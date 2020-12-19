"""
TODO: docstring

"""

import gc
import json
from pathlib import Path
from typing import List, Optional, Tuple, Union, Sequence
import warnings

import faiss
import numpy as np

from interactive_index.config import read_config, CONFIG_DEFAULTS
from interactive_index.utils import (merge_on_disk,
                                     to_all_gpus,
                                     cantor_pairing,
                                     invert_cantor_pairing)


# Replicates Path.unlink(missing_ok=True) functionality that was added in Python 3.8;
# we want to preserve compatibility with Python 3.7
def _unlink(path: Path) -> None:
    try:
        path.unlink()
    except FileNotFoundError:
        pass


class InteractiveIndex:
    """
    This class is a sharded, on-disk index that can be iteratively trained,
    added to, merged, and searched.

    The vectors are stored in partial (or shard) indexes that are in separate
    files. When `merge_partial_indexes()` is called, the indexes are merged
    on disk, and the underlying data is extracted to a separate ".ivfdata"
    file. The search is done on this merged index without  having to load all
    the underlying data into memory.

    TODO: explain construction, attributes

    """

    TRAINED_INDEX_NAME = 'trained.index'
    SHARD_INDEX_NAME_TMPL = 'shard_{}.index'
    MERGED_INDEX_DATA_NAME = 'merged.ivfdata'
    MERGED_INDEX_NAME = 'merged.index'
    META_FILE_NAME = 'meta.json'

    SUPPORTED_DIM_PER_SUBQ = [1, 2, 3, 4, 6, 8, 10, 12, 16, 20, 24, 28, 32]

    _invert_cantor_pairing_vec = np.vectorize(invert_cantor_pairing)

    def __init__(self, config_fpath: str = None, **kwargs) -> None:
        """
        TODO: explain usage, kwargs

        """

        exists = False
        if '_exists' in kwargs:
            exists = kwargs['_exists']
            del kwargs['_exists']

        extra = None
        if '_extra' in kwargs:
            extra = kwargs['_extra']
            del kwargs['_extra']

        assert not exists or extra

        if config_fpath:
            self.cfg = read_config(config_fpath)
        else:
            self.cfg = CONFIG_DEFAULTS.copy()

        self.cfg.update(kwargs)

        self.d = self.cfg['d']
        self.n_centroids = self.cfg['n_centroids']
        self.n_probes = self.cfg['n_probes']
        self.vectors_per_index = self.cfg['vectors_per_index']

        self.tempdir = Path(self.cfg['tempdir'])
        self.tempdir.mkdir(parents=True, exist_ok=True)

        self.use_gpu = self.cfg['use_gpu']
        self.train_on_gpu = self.cfg['train_on_gpu'] or self.use_gpu
        self.use_float16 = self.cfg['use_float16']
        self.use_float16_quantizer = self.cfg['use_float16_quantizer']
        self.use_precomputed_codes = self.cfg['use_precomputed_codes']

        self.search = self.cfg['search']
        self.search_args = self.cfg['search_args']
        self.transform = self.cfg['transform']
        self.transform_args = self.cfg['transform_args']
        self.encoding = self.cfg['encoding']
        self.encoding_args = self.cfg['encoding_args']

        self.index_str = self._create_index_str(
            self.n_centroids,
            self.search, self.search_args,
            self.encoding, self.encoding_args,
            self.transform, self.transform_args
        )
        self.metric = self.cfg['metric'].lower()

        if self.use_gpu:
            self.co = self._create_co(
                self.use_float16,
                self.use_float16_quantizer,
                self.use_precomputed_codes
            )
        else:
            self.co = None

        self.multi_id = self.cfg['multi_id']

        if not exists:
            self.create(self.index_str)
        else:
            self.requires_training = extra['requires_training']
            self.is_trained = extra['is_trained']
            self.n_indexes = extra['n_indexes']
            self.n_vectors = extra['n_vectors']

        self._save_metadata()

    def create(self, index_str: str) -> None:
        """
        Creates the empty index specified by index_str and writes to disk.

        Args:
            index_str: The FAISS index factory string specification.

        """

        metric = faiss.METRIC_L2
        if self.metric == 'inner product':
            metric = faiss.METRIC_INNER_PRODUCT

        index = faiss.index_factory(self.d, index_str, metric)
        faiss.write_index(index, str(self.tempdir/self.TRAINED_INDEX_NAME))

        # TODO: get from index_str
        self.requires_training = True
        self.is_trained = False
        self.n_indexes = 0
        self.n_vectors = 0

    def train(self, xt_src: Union[str, np.ndarray, List[float]]) -> None:
        """
        Trains the clustering layer of the index.

        Args:
            xt_src: The source of the training vectors. Can be the name of a
                file output by np.ndarray.tofile(), a numpy array, or a list
                of floats.

        """

        if not self.requires_training:
            warnings.warn(
                '`train()` was called on a non-trainable index.',
                RuntimeWarning
            )
            return

        xt = self._convert_src_to_numpy(xt_src)

        index = faiss.read_index(str(self.tempdir/self.TRAINED_INDEX_NAME))

        if self.train_on_gpu:
            # TODO: perhaps add memory check for training on GPU?
            index = to_all_gpus(index, self.co)

        index.train(xt)
        index.reset()

        faiss.write_index(
            faiss.index_gpu_to_cpu(index) if self.train_on_gpu else index,
            str(self.tempdir/self.TRAINED_INDEX_NAME)
        )

        self.is_trained = True
        self._save_metadata()

    def add(
        self,
        xb_src: Union[str, np.ndarray, List[float]],
        ids: Optional[Sequence[int]] = None,
        ids_extra: Optional[Union[Sequence[int], int]] = 0,
        update_metadata: bool = True
    ) -> None:
        """
        Adds the given vectors to the index.

        Args:
            xb_src: The source of the vectors to add to the partial indexes.
                Can be the name of a file output by np.ndarray.tofile(), a
                numpy array, or a list of floats.
            ids: The integer IDs used to identify each vector added. If not
                specified will just increment sequentially starting from the
                the number of vectors in the index.
            ids_extra: The extra id field if multi_id is True. This allows you
                to identify a vector through two different fields.

                E.g., you might have two embeddings for each image, so you
                could make the ID the image number, and the ID extra a 0 or 1
                for each embedding.
            update_metadata: Whether to update the metadata JSON file after
                adding with the new number of vectors and shards this index
                contains.
        """

        if not self.is_trained:
            raise RuntimeError('Cannot add to untrained index.')

        xb = self._convert_src_to_numpy(xb_src)
        if ids is None:
            ids = np.arange(self.n_vectors, self.n_vectors + len(xb))
        else:
            ids = np.array(ids)

        if self.multi_id:
            if isinstance(ids_extra, int):
                ids_extra = np.full(len(ids), ids_extra)
            else:
                ids_extra = np.array(ids_extra)

            ids = np.array([
                cantor_pairing(ids[i], ids_extra[i]) for i in range(len(ids))
            ])

        idx = 0
        while idx < len(xb):
            # gc.collect()

            if self.n_vectors % self.vectors_per_index == 0:
                # Need to create a new index
                index = faiss.read_index(
                    str(self.tempdir/self.TRAINED_INDEX_NAME)
                )
                shard_num = self.n_indexes
                self.n_indexes += 1
            else:
                # Read existing partial index
                shard_num = self.n_indexes - 1
                index = faiss.read_index(str(
                    self.tempdir/self.SHARD_INDEX_NAME_TMPL.format(shard_num)
                ))

            end_idx = idx + min(
                self.vectors_per_index - (self.n_vectors % self.vectors_per_index),
                xb.shape[0]
            )

            index.add_with_ids(xb[idx:end_idx], ids[idx:end_idx])
            self.n_vectors += (end_idx - idx)

            faiss.write_index(
                index,
                str(self.tempdir/self.SHARD_INDEX_NAME_TMPL.format(shard_num))
            )

            index.reset()
            # del index  # force free memory

            if update_metadata:
                self._save_metadata()

            idx = end_idx

    def merge_partial_indexes(
        self,
        shard_index_names: Optional[List[str]] = None
    ) -> None:
        """
        TODO: docstring

        """

        index = faiss.read_index(str(self.tempdir/self.TRAINED_INDEX_NAME))

        shard_index_names = shard_index_names or [
            str(self.tempdir/self.SHARD_INDEX_NAME_TMPL.format(shard_num))
            for shard_num in range(self.n_indexes)
        ]

        merge_on_disk(
            index,
            shard_index_names,
            str(self.tempdir/self.MERGED_INDEX_DATA_NAME)
        )

        faiss.write_index(index, str(self.tempdir/self.MERGED_INDEX_NAME))

    def query(
        self,
        xq_src: Union[str, np.ndarray, List[float]],
        k: int = 1,
        n_probes: Optional[int] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        TODO: docstring

        """

        xq = self._convert_src_to_numpy(xq_src)
        index = faiss.read_index(str(self.tempdir/self.MERGED_INDEX_NAME))
#         if self.use_gpu:
#             # TODO: perhaps add memory check for training on GPU?
#             index = to_all_gpus(index, self.co)
        index_ivf = faiss.extract_index_ivf(index)
        index_ivf.nprobe = n_probes if n_probes else self.n_probes
        dists, inds = index.search(xq, k)

        if self.multi_id:
            inds = self._invert_cantor_pairing_vec(inds)

        return dists, inds

    def cleanup(self) -> None:
        """Deletes all persistent files associated with this index."""

        _unlink(self.tempdir/self.TRAINED_INDEX_NAME)
        _unlink(self.tempdir/self.MERGED_INDEX_NAME)
        _unlink(self.tempdir/self.MERGED_INDEX_DATA_NAME)
        _unlink(self.tempdir/self.META_FILE_NAME)
        self.delete_shards()

    def delete_shards(self) -> None:
        """
        Deletes all shard indexes. merge_partial_indexes() and add() will
        fail after calling.

        """

        for shard_num in range(self.n_indexes):
            _unlink(self.tempdir/self.SHARD_INDEX_NAME_TMPL.format(shard_num))

    #####################
    # Private Functions #
    #####################

    def _convert_src_to_numpy(
        self,
        x_src: Union[str, np.ndarray, List[float]]
    ) -> np.ndarray:
        """
        TODO: docstring

        """

        if isinstance(x_src, str):
            x = np.fromfile(x_src, dtype='float32').reshape(-1, self.d).copy()
        elif isinstance(x_src, np.ndarray):
            x = np.ascontiguousarray(x_src.reshape(-1, self.d))
        elif isinstance(x_src, list):
            x = np.array(x_src, size=(-1, self.d), ndmin=2, dtype='float32')
        else:
            raise TypeError(
                f"Input of type '{type(x_src)}' not supported. Use a str (the "
                 "file path), a numpy.ndarray, or a list of floats instead."
            )

        return x

    def _create_co(
        self,
        use_float16,
        use_float16_quantizer,
        use_precomputed_codes
    ) -> 'faiss.GpuMultipleClonerOptions':
        """
        TODO: docstring

        """

        co = faiss.GpuMultipleClonerOptions()
        co.useFloat16 = self.use_float16
        co.useFloat16CoarseQuantizer = self.use_float16_quantizer
        co.usePrecomputed = use_precomputed_codes
        return co

    def _create_index_str(
        self,
        n_centroids: int,
        search: Optional[str],
        search_args: Optional[List],
        encoding: Optional[str],
        encoding_args: Optional[List],
        transform: Optional[str],
        transform_args: Optional[List]
    ) -> str:
        """
        TODO: docstring

        """

        # Search
        search_str_parts = [f'IVF{n_centroids}'] if n_centroids else []
        if search:
            if search_args:
                assert all(isinstance(arg, int) for arg in search_args)
                search += 'x'.join(str(arg) for arg in search_args)
            search_str_parts.append(search)

        search_str = '_'.join(search_str_parts)

        # Transformation
        transform_str = transform if transform else ''
        if transform_args:
            if transform in ['PCA', 'PCAR', 'ITQ', 'OPQ']:
                assert all(isinstance(arg, int) for arg in transform_args)

                transform_str += '_'.join(str(arg) for arg in transform_args)

        # Encoding
        encoding_str = encoding
        if encoding_args:
            if encoding in ['SQ', 'PQ']:
                assert all(isinstance(arg, int) for arg in encoding_args)

                encoding_str += '+'.join(str(arg) for arg in encoding_args)

                if encoding == 'PQ':
                    dim_per_subq = self.d / encoding_args[0]
                    if dim_per_subq not in self.SUPPORTED_DIM_PER_SUBQ:
                        self.use_precomputed_codes = True

            elif encoding in ['LSH']:
                assert isinstance(encoding_args[0], str)

                if encoding_args[0] == 'rotate':
                    encoding_str += 'r'

        return ','.join([transform_str, search_str, encoding_str])

    def _save_metadata(self) -> None:
        """
        TODO: docstring

        """

        payload = {
            'cfg': self.cfg,
            'extra': {
                'requires_training': self.requires_training,
                'is_trained': self.is_trained,
                'n_indexes': self.n_indexes,
                'n_vectors': self.n_vectors
            }
        }

        json.dump(payload, (self.tempdir/self.META_FILE_NAME).open('w'))

    @classmethod
    def load(cls, tempdir: str) -> 'InteractiveIndex':
        """
        TODO: docstring

        """

        payload = json.load((Path(tempdir)/cls.META_FILE_NAME).open())
        index = InteractiveIndex(
            **payload['cfg'], _extra=payload['extra'], _exists=True
        )
        return index

