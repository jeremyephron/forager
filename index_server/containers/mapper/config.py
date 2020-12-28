from detectron2.config.config import get_cfg as get_default_detectron_config
import os

WEIGHTS_PATH = "R-50.pkl"  # model will be downloaded here during container build
RESNET_CONFIG = get_default_detectron_config()
RESNET_CONFIG.MODEL.RESNETS.OUT_FEATURES = ["res5"]

EMBEDDINGS_FILE_TMPL = "/shared/embeddings/{}/{}/{}.npy"
GCS_URL_PREFIX = "https://storage.googleapis.com"
DOWNLOAD_NUM_RETRIES = 4

NUM_CPUS = int(os.getenv("CPUS", "1"))
