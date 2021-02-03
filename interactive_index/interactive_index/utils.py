"""
File: utils.py
--------------
This module contains utility functions for the interactive index. These
functions should likely not be called manually.

"""

import math
from typing import List, Optional, Tuple

import faiss
import numpy as np
from sklearn.metrics import pairwise_distances


# Units in bytes
BYTE     = BYTES     = 2**0
KILOBYTE = KILOBYTES = 2**10
MEGABYTE = MEGABYTES = 2**20
GIGABYTE = GIGABYTES = 2**30
TERABYTE = TERABYTES = 2**40


def round_down_to_pow2(x: int) -> int:
    if x == 0:
        return 0

    return 1 << (x.bit_length() - 1)


def round_up_to_pow2(x: int) -> int:
    if x == 0:
        return 0

    return 1 << ((x - 1).bit_length())


def round_down_to_mult(x: int, m: int) -> int:
    return x // m * m


def round_up_to_mult(x: int, m: int) -> int:
    return (x + m - 1) // m * m


def merge_on_disk(
	trained_index: faiss.Index,
    shard_fnames: List[str],
    ivfdata_fname: str
) -> None:
    """
	Adds the contents of the indexes stored in shard_fnames into the index
    trained_index. The on-disk data is stored in ivfdata_fname.

    Args:
        trained_index: The trained index to add the data to.
        shard_fnames: A list of the partial index filenames.
        ivfdata_fname: The filename for the on-disk extracted data.

	"""

    # Load the inverted lists
    ivfs = []
    for fname in shard_fnames:
        # The IO_FLAG_MMAP is to avoid actually loading the data, and thus the
        # total size of the inverted lists can exceed the available RAM
        index = faiss.read_index(fname, faiss.IO_FLAG_MMAP)
        index_ivf = faiss.extract_index_ivf(index)
        ivfs.append(index_ivf.invlists)

        # Avoid deallocating the invlists with the index
        index_ivf.own_invlists = False

    # Construct the output index
    index = trained_index
    index_ivf = faiss.extract_index_ivf(index)

    assert index.ntotal == 0, 'The trained index should be empty'

    # Prepare the output inverted lists, which are written to ivfdata_fname.
    invlists = faiss.OnDiskInvertedLists(index_ivf.nlist, index_ivf.code_size,
                                         ivfdata_fname)

    # Merge all the inverted lists
    ivf_vector = faiss.InvertedListsPtrVector()
    for ivf in ivfs:
        ivf_vector.push_back(ivf)

    n_total = invlists.merge_from(ivf_vector.data(), ivf_vector.size())

    # Replace the inverted lists in the output index
    index.ntotal = index_ivf.ntotal = n_total
    index_ivf.replace_invlists(invlists, True)
    invlists.this.disown()


def to_all_gpus(
    cpu_index: faiss.Index,
    co: Optional['faiss.GpuMultipleClonerOptions'] = None
) -> faiss.Index:
    """
    TODO: docstring

    """

    n_gpus = faiss.get_num_gpus()
    assert n_gpus != 0, 'Attempting to move index to GPU without any GPUs'

    gpu_index = faiss.index_cpu_to_all_gpus(cpu_index, co=co)
    return gpu_index


def cantor_pairing(a: int, b: int) -> int:
    """
    TODO: docstring

    """

    c = a + b
    return (c * (c + 1)) // 2 + b


def invert_cantor_pairing(z: int) -> Tuple[int, int]:
    """
    TODO: docstring

    """

    w = math.floor((math.sqrt(8 * z + 1) - 1) / 2)
    t = (w**2 + w) / 2
    y = int(z - t)
    x = int(w - y)
    return x, y


def sample_farthest_vectors(
    index: 'InteractiveIndex',
    xq: np.ndarray,
    percent: float,
    n_samples: int
) -> np.ndarray:
    """
    Samples a percentage of the furthest vectors (by cluster distance) to a
    query vector.

    Args:
        index: The index to search within.
        xq: The query vector.
        percent: The bottom percentage of vectors to sample from.
        n_samples: The maximum number of vectors to sample.

    Returns:
        The IDs of the vectors.

    """

    centroids = index.get_centroids()
    xq = index.apply_transform(xq)

    if index.metric == 'inner product':
        farthest_centroid_inds = np.argsort(
            [-xq.dot(centroids[i]) for i in range(len(centroids))]
        )
    else:
        farthest_centroid_inds = np.argsort(
            [-np.linalg.norm(xq - centroids[i]) for i in range(len(centroids))]
        )

    total = 0
    cluster_inds_to_use = []
    cluster_sizes = index.get_cluster_sizes()
    for i in farthest_centroid_inds:
        total += cluster_sizes[i]
        cluster_inds_to_use.append(i)
        if total >= index.n_vectors * percent:
            break

    sample_ids = []
    sample_extra_ids = []
    for i in cluster_inds_to_use:
        ids = index.get_cluster_ids(farthest_centroid_inds[i])
        if index.multi_id:
            ids, extra_ids = ids
        else:
            extra_ids = None

        print(len(ids))
        random_inds = np.random.choice(
            len(ids),
            size=min(len(ids), int(n_samples * cluster_sizes[i] / total)),
            replace=False,
        )

        sample_ids.append(ids[random_inds])
        if extra_ids is not None:
            sample_extra_ids.append(extra_ids[random_inds])

    if index.multi_id:
        return np.concatenate(sample_ids), np.concatenate(sample_extra_ids)
    else:
        return np.concatenate(sample_ids)
