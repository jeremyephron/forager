"""
File: config.py
---------------
Contains config defaults for the InteractiveIndex, and functions for creating
config dictionaries.

"""

import copy
import math

import yaml

from interactive_index.utils import *


CONFIG_DEFAULTS = {
    'd': 1_024,
    'n_centroids': 32, # only used with IVF
    'n_probes': 16, # only used with IVF
    'vectors_per_index': 10_000,
    'tempdir': '/tmp/',
    'use_gpu': False,
    'train_on_gpu': False,
    'use_float16': False,
    'use_float16_quantizer': False,
    'use_precomputed_codes': False,
    'metric': 'L2',

    # Transformation
    'transform': None, # {'PCA', 'PCAR', 'ITQ', 'OPQ'}
    'transform_args': None,

    # Search
    'search': None, # {'HNSW'}
    'search_args': None,

    # Encoding
    'encoding': 'Flat', # {'SQ', 'PQ', 'LSH'}
    'encoding_args': None,

    # Misc
    'multi_id': False,
    'direct_map': 'NoMap', # {'NoMap', 'Array', 'Hashtable'}
}


def read_config(config_fpath: str) -> dict:
    """
    Loads a config dictionary from the specified yaml file.

    Args:
        config_fpath: The file path of the config yaml file.

    Returns:
        The dictionary of the configuration options, with default options if
        any were missing.

    """

    cfg = yaml.load(open(config_fpath, 'r'), Loader=yaml.FullLoader)

    # Set defaults if missing
    for key, val in CONFIG_FILE.items():
        if key not in cfg:
            cfg[key] = val

    return cfg


def _set_config_for_mem_usage(d: int, n_vecs: int, max_mem: int, config: dict):
    """

    Args:
        d: The vector dimension.
        n_vecs: The expected number of vectors.
        max_mem: The maximum amount of memory you want the
            index to take up in bytes.
        config: The configuration dictionary to set.

    """

    FLOAT32_SZ = 4 * BYTES
    MAX_SUBINDEX_SZ = 1 * GIGABYTE

    # Flat
    if n_vecs * d * FLOAT32_SZ <= max_mem:
        config['encoding'] = 'Flat'
        config['vectors_per_index'] = MAX_SUBINDEX_SZ // (d * FLOAT32_SZ)

    # Scalar Quantization to 1 byte
    elif n_vecs * d <= max_mem:
        config['encoding'] = 'SQ'
        config['encoding_args'] = [8]
        config['vectors_per_index'] = MAX_SUBINDEX_SZ // d

    else:
        # PCA target dim or PQ subvectors
        x = round_down_to_mult(max_mem // n_vecs, 4)

        # PCA with rotation to at least 64 dimensions and
        # Scalar Quantization to 1 byte
        if x > 64:
            config['transform'] = 'PCAR'
            config['transform_args'] = [x]

            config['encoding'] = 'SQ'
            config['encoding_args'] = [8]

            config['vectors_per_index'] = MAX_SUBINDEX_SZ // x

        # OPQ with PQ
        else:
            y = max(filter(
                lambda k: k <= 4 * x and x <= d,
                [x, 2 * x, 3 * x, 4 * x, d]
            ))

            config['transform'] = 'OPQ'
            config['transform_args'] = [x, y]

            config['encoding'] = 'PQ'
            config['encoding_args'] = [x]

            config['vectors_per_index'] = MAX_SUBINDEX_SZ // x


def auto_config(d: int, n_vecs: int, max_ram: int, pca_d: int = None, sq: int = None):
    """

    """

    FLOAT32_SZ = 4 * BYTES

    config = copy.copy(CONFIG_DEFAULTS)
    config['d'] = d

    if pca_d:
        config['transform'] = 'PCAR'
        config['transform_args'] = [pca_d]
    if sq:
        config['encoding'] = 'SQ'
        config['encoding_args'] = [sq]

#     _set_config_for_mem_usage(d, n_vecs, max_mem, config)

#     if n_vectors < 10_000:
#         "HNSW32"
#         memory_usage = d * 4 + 32 * 2 * 4

    if n_vecs < 1_000_000:
        # IVFx
        n_centroids = round_up_to_pow2(4 * int(math.sqrt(n_vecs)))
        if n_centroids > 16 * math.sqrt(n_vecs):
            n_centroids = round_up_to_mult(4 * int(math.sqrt(n_vecs)), 4)

        # train needs to be [30*n_centroids, 256*n_centroids]
        config['n_centroids'] = n_centroids
        config['recommended_n_train'] = n_centroids * 39
        config['n_probes'] = n_centroids // 16
        return config

    config['use_gpu'] = False
    #config['search'] = 'HNSW'
    #config['search_args'] = [32]

    if n_vecs < 10_000_000:
        # IVF65536_HNSW32
        # not supported on GPU, if need GPU use IVFx
        # can train on GPU though

        # Want 2**16, but RAM problem
        for i in range(12, 3, -1):
            n_centroids = 2**i
            if max_ram / (FLOAT32_SZ * d) > n_centroids * 39:
                config['n_centroids'] = n_centroids
                config['recommended_n_train'] = n_centroids * 39
                config['n_probes'] = max(n_centroids // 16, 1)
                return config

        assert False, 'Too little RAM'

    elif n_vecs < 100_000_000:
        # IVF262144_HNSW32
        # not supported on GPU, if need GPU use IVFx
        # can train on GPU though

        # Want 2**18, but RAM problem
        for i in range(12, 5, -1):
            n_centroids = 2**i
            if max_ram / (FLOAT32_SZ * d) > n_centroids * 39:
                config['n_centroids'] = n_centroids
                config['recommended_n_train'] = n_centroids * 39
                config['n_probes'] = max(n_centroids // 16, 1)
                return config

        assert False, 'Too little RAM'

    else: # good for n_vectors < 1_000_000_000
        # IVF1048576_HNSW32
        # not supported on GPU, if need GPU use IVFx
        # can train on GPU though

        # Want 2**20, but RAM problem
        for i in range(13, 5, -1):
            n_centroids = 2**i
            if max_ram / (FLOAT32_SZ * d) > n_centroids * 39:
                config['n_centroids'] = n_centroids
                config['recommended_n_train'] = n_centroids * 39
                config['n_probes'] = max(n_centroids // 8, 1)
                return config

        assert False, 'Too little RAM'

