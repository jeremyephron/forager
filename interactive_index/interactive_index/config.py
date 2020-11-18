"""
File: config.py
---------------
Contains config defaults for the InteractiveIndex, and functions for creating
config dictionaries.

"""

import copy
import os

import yaml

from interactive_index.utils import *


CONFIG_DEFAULTS = {
    'd': 1_024,
    'n_centroids': 32, # only used with IVF
    'n_probes': 4, # only used with IVF
    'vectors_per_index': 10_000,
    'tempdir': '/tmp/',
    'use_gpu': False,
    'use_float16': False,
    'use_float16_quantizer': False,
    'use_precomputed_codes': False,
    'metric': 'L2',
    
    # Transformation
    'transform': None, # {'PCA', 'PCAR', 'ITQ', 'OPQ'}
    'transform_args': None,
    
    # Search
    'search': None # {'HNSW'}
    'search_args': None

    # Encoding
    'encoding': 'Flat', # {'SQ', 'PQ', 'LSH'}
    'encoding_args': None,

    # Misc
    'multi_id': False
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
        target_mem: The desired amount of memory you want 
            the index to take up in bytes.
    
    """
    
    # Determine optimal memory balance
    
    # Flat
    if n_vecs * d * FLOAT32_SZ <= max_mem:
        config['encoding'] = 'Flat'
    
    # Scalar Quantization to 1 byte
    elif n_vecs * d <= max_mem:
        config['encoding'] = 'SQ'
        config['encoding_args'] = [8]
    
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
            
        # OPQ with PQ
        else:
            y = max(filter(
                lambda k: k <= 4 * x and x <= d,
                [x, 2 * x, 3 * x, 4 * x, d]
            ))
            
            config['transform'] = 'OPQ'
            config['transform_args'] = [x, y]
            
            config['encoding'] = 'PQ'
            config['encoding_args'] = x
    

def auto_config(d: int, n_vecs: int, max_mem: int, target_mem: int):
    """
    Automatically suggests a configuration given some information.
    
    Args:
        d: The vector dimension.
        n_vecs: The expected number of vectors.
        max_mem: The maximum amount of memory you want the 
            index to take up in bytes.
        target_mem: The desired amount of memory you want 
            the index to take up in bytes.
        
    """
    
    config = copy.copy(CONFIG_DEFAULTS)
    config['d'] = d
    config
#     {
#     'd': 1_024,
#     'n_centroids': 32,
#     'n_probes': 4,
#     'vectors_per_index': 10_000,
#     'tempdir': '/tmp/',
#     'use_gpu': True,
#     'use_float16': True,
#     'use_float16_quantizer': True,
#     'use_precomputed_codes': False,
#     'metric': 'L2',
    
#     # Transformation
#     'transform': None,
#     'transform_args': None,

#     # Encoding
#     'encoding': 'Flat',
#     'encoding_args': None,

#     # Misc
#     'multi_id': False
# }
    
    FLOAT32_SZ = 4  # bytes
    

    _set_config_for_mem_usage(d, n_vecs, max_mem, config)
    
#     if n_vectors < 10_000:
#         "HNSW32"
#         memory_usage = d * 4 + 32 * 2 * 4
    
    if n_vectors < 1_000_000:
        "...,IVFx,..."
        # where 4*sqrt(n) <= x <= 16*sqrt(n)
        # train needs to be [30*x, 256*x]
        
    elif n_vectors < 10_000_000:
        "...,IVF65536_HNSW32,..."
        # not supported on GPU, if need GPU use IVFx
    
    elif n_vectors < 100_000_000:
        "...,IVF262144_HNSW32,..."
        # can train on GPU
        
    else: # good for n_vectors < 1_000_000_000
        "...,IVF1048576_HNSW32,..."
        