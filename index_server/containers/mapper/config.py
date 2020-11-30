from detectron2.config.config import get_cfg as get_default_detectron_config

WEIGHTS_PATH = "R-50.pkl"  # model will be downloaded here during container build
RESNET_CONFIG = get_default_detectron_config()
RESNET_CONFIG.MODEL.RESNETS.OUT_FEATURES = ["res5"]

EMBEDDINGS_FILE_PATTERN = "/shared/embeddings/{}/{}/{}.npy"
