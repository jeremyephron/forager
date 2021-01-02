from detectron2.config.config import get_cfg as get_default_detectron_config
import functools
import numpy as np
import os

WEIGHTS_PATH = "R-50.pkl"  # model will be downloaded here during container build
RESNET_CONFIG = get_default_detectron_config()
RESNET_CONFIG.MODEL.RESNETS.OUT_FEATURES = ["res5"]

GCS_URL_PREFIX = "https://storage.googleapis.com"
DOWNLOAD_NUM_RETRIES = 4

EMBEDDINGS_FILE_TMPL = "/shared/embeddings/{}/{}/{}-{{}}.npy"
REDUCTIONS = {
    "none": lambda x: x,
    "average": functools.partial(np.mean, axis=0, keepdims=True),
}

NPROC = int(os.getenv("NPROC", "1"))
