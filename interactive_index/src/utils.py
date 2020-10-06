"""
File: utils.py
--------------
This module contains utility functions for the interactive index. These 
functions should likely not be called manually.

"""

import math
from typing import List, Optional, Tuple

import faiss


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
    co: Optional[faiss.GpuMultipleClonerOptions] = None
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
