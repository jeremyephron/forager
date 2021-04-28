import concurrent.futures
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

from detectron2.layers import ShapeSpec
from detectron2.checkpoint.detection_checkpoint import DetectionCheckpointer
from detectron2.modeling.backbone.resnet import build_resnet_backbone
from detectron2.config.config import get_cfg as get_default_detectron_config

BATCH_SIZE = 8
IMAGE_DIR = Path("/home/mihir/waymo/val")
EMBEDDING_DIMS = {"res4": 1024, "res5": 2048}

WEIGHTS_PATH = "R-50.pkl"
RESNET_CONFIG = get_default_detectron_config()
RESNET_CONFIG.MODEL.RESNETS.OUT_FEATURES = list(EMBEDDING_DIMS.keys())

IMAGE_LIST_OUTPUT_FILENAME = "images.json"
EMBEDDINGS_OUTPUT_FILENAME_TMPL = "embeddings_{}.npy"

# Create model
shape = ShapeSpec(channels=3)
model = torch.nn.Sequential(build_resnet_backbone(RESNET_CONFIG, shape))

# Load model
device = "cuda" if torch.cuda.is_available() else "cpu"
checkpointer = DetectionCheckpointer(model, save_to_disk=False)
checkpointer.load(WEIGHTS_PATH)
model.to(device)
model.eval()

# Load image paths
image_paths = list(IMAGE_DIR.glob("*.jpeg"))
json.dump([str(p) for p in image_paths], Path(IMAGE_LIST_OUTPUT_FILENAME).open("w"))

embeddings = {
    layer: np.memmap(
        EMBEDDINGS_OUTPUT_FILENAME_TMPL.format(layer),
        dtype="float32",
        mode="w+",
        shape=(len(image_paths), dim),
    )
    for layer, dim in EMBEDDING_DIMS.items()
}

pixel_mean = torch.tensor(RESNET_CONFIG.MODEL.PIXEL_MEAN).view(-1, 1, 1)
pixel_std = torch.tensor(RESNET_CONFIG.MODEL.PIXEL_STD).view(-1, 1, 1)
input_format = RESNET_CONFIG.INPUT.FORMAT


def load_image(path):
    image = Image.open(path)
    image = image.convert("RGB")
    image = torch.as_tensor(np.asarray(image), dtype=torch.float32)  # -> tensor
    image = image.permute(2, 0, 1)  # HWC -> CHW
    if input_format == "BGR":
        image = torch.flip(image, dims=(0,))  # RGB -> BGR
    image = image.contiguous()
    image = (image - pixel_mean) / pixel_std
    image = image.unsqueeze(dim=0)
    return image.to(device)


with concurrent.futures.ThreadPoolExecutor() as executor:
    for i in tqdm(range(0, len(image_paths), BATCH_SIZE)):
        # Load batch of images
        batch_paths = image_paths[i : i + BATCH_SIZE]
        images = torch.cat(list(executor.map(load_image, batch_paths)))

        with torch.no_grad():
            output_dict = model(images)
            for layer, e in embeddings.items():
                outs = output_dict[layer].mean(dim=(2, 3))
                e[i : i + BATCH_SIZE] = outs.cpu().numpy()

for e in embeddings.values():
    e.flush()
