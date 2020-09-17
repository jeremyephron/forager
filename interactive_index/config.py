"""
File: config.py
---------------
Contains config defaults for the InteractiveIndex, and functions for creating
config dictionaries.

"""

import os

import yaml


CONFIG_DEFAULTS = {
    'd': 1_024,
    'n_centroids': 32,
    'n_probes': 4,
    'vectors_per_index': 10_000,
    'tempdir': '/tmp/',
    'use_gpu': True,
    'use_float16': True,
    'use_float16_quantizer': True,
    'use_precomputed_codes': False,
    
    # Transformation
    'transform': None,
    'transform_args': None,

    # Encoding
    'encoding': 'Flat',
    'encoding_args': None
}


def read_config(config_fpath: str) -> dict:
    """
    Loads a config dictionary from the specified yaml file.

    Args:
        config_fpath: the file path of the config yaml file.

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
