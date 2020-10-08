"""
TODO: docstring

"""

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

    SUPPORTED_DIM_PER_SUBQ = [1, 2, 3, 4, 6, 8, 10, 12, 16, 20, 24, 28, 32]

    _invert_cantor_pairing_vec = np.vectorize(invert_cantor_pairing)
    
    def __init__(self, config_fpath: str = None, **kwargs) -> None:
        """
        TODO: explain usage, kwargs

        """

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
        self.use_float16 = self.cfg['use_float16']
        self.use_float16_quantizer = self.cfg['use_float16_quantizer']
        self.use_precomputed_codes = self.cfg['use_precomputed_codes']
        
        self.transform = self.cfg['transform']
        self.transform_args = self.cfg['transform_args']
        self.encoding = self.cfg['encoding']
        self.encoding_args = self.cfg['encoding_args']
        
        self.index_str = self._create_index_str(
            self.n_centroids,
            self.encoding, self.encoding_args, 
            self.transform, self.transform_args
        )
        
        if self.use_gpu:
            self.co = self._create_co(
                self.use_float16,
                self.use_float16_quantizer,
                self.use_precomputed_codes
            )
        else:
            self.co = None

        self.multi_id = self.cfg['multi_id']

        self.create(self.index_str)
    
    def create(self, index_str: str) -> None:
        """
        Creates the empty index specified by index_str and writes to disk.

        Args:
            index_str: The FAISS index factory string specification.

        """

        index = faiss.index_factory(self.d, index_str)
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

        if self.use_gpu:
            # TODO: perhaps add memory check for training on GPU?
            index = to_all_gpus(index, self.co)

        index.train(xt)

        faiss.write_index(
            faiss.index_gpu_to_cpu(index) if self.use_gpu else index,
            str(self.tempdir/self.TRAINED_INDEX_NAME)
        )

        self.is_trained = True

    def add(
        self,
        xb_src: Union[str, np.ndarray, List[float]],
        ids: Optional[Sequence[int]] = None,
        ids_extra: Optional[Union[Sequence[int], int]] = 0
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

        """

        if not self.is_trained:
            raise RuntimeError('Cannot add to untrained index.')
        
        xb = self._convert_src_to_numpy(xb_src)

        if self.n_vectors % self.vectors_per_index == 0:
            # Need to create a new index
            index = faiss.read_index(str(self.tempdir/self.TRAINED_INDEX_NAME))
            shard_num = self.n_indexes
            self.n_indexes += 1
        else:
            # Read existing partial index
            shard_num = self.n_indexes - 1
            index = faiss.read_index(str(
                self.tempdir/self.SHARD_INDEX_NAME_TMPL.format(shard_num)
            ))

        # Move to GPU
        if self.use_gpu:
            index = to_all_gpus(index, self.co)

        end_idx = min(
            self.vectors_per_index - (self.n_vectors % self.vectors_per_index),
            xb.shape[0]
        )
        
        if ids is None:
            ids = np.arange(self.n_vectors, self.n_vectors + end_idx)
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

        index.add_with_ids(xb[:end_idx], ids)
        self.n_vectors += end_idx

        faiss.write_index(
            faiss.index_gpu_to_cpu(index), 
            str(self.tempdir/self.SHARD_INDEX_NAME_TMPL.format(shard_num))
        )

        if end_idx < xb.shape[0]:
           self.add(xb[end_idx:])

    def merge_partial_indexes(self) -> None:
        """
        TODO: docstring

        """

        index = faiss.read_index(str(self.tempdir/self.TRAINED_INDEX_NAME))

        shard_index_names = [
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
        index.nprobe = n_probes if n_probes else self.n_probes
        dists, inds = index.search(xq, k)
        
        if self.multi_id:
            inds = self._invert_cantor_pairing_vec(inds)

        return dists, inds

    def cleanup() -> None:
        """Deletes all persistent files associated with this index."""

        (self.tempdir/self.TRAINED_INDEX_NAME).unlink(missing_ok=True)
        (self.tempdir/self.MERGED_INDEX_NAME).unlink(missing_ok=True)
        (self.tempdir/self.MERGED_INDEX_DATA_NAME).unlink(missing_ok=True)
        for shard_num in range(self.n_indexes):
            (self.tempdir/self.SHARD_INDEX_NAME_TMPL.format(shard_num)).unlink(
                missing_ok=True
            )

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
            x = x_src.reshape(-1, self.d).copy()
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
    ) -> faiss.GpuMultipleClonerOptions:
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
        encoding: str,
        encoding_args: Optional[List],
        transform: Optional[str],
        transform_args: Optional[List]
    ) -> str:
        """
        TODO: docstring

        """

        transform_str = transform if transform else ''
        if transform_args:
            if transform in ['PCA', 'ITQ']:
                assert isinstance(transform_args[0], int)

                transform_str += str(transform_args[0])
            
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

        search_str = f'IVF{n_centroids}' 
       
        return ','.join([transform_str, search_str, encoding_str])

