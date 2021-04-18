from pathlib import Path
import json

import numpy as np
import torch
import clip
from PIL import Image
from tqdm import tqdm

BATCH_SIZE = 128
IMAGE_DIR = Path("/home/mihir/waymo/train")
EMBEDDING_DIM = 512

IMAGE_LIST_OUTPUT_FILENAME = "images.json"
EMBEDDINGS_OUTPUT_FILENAME = "embeddings.npy"

device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)

image_paths = list(IMAGE_DIR.glob("*.jpeg"))
json.dump([str(p) for p in image_paths], Path(IMAGE_LIST_OUTPUT_FILENAME).open("w"))

embeddings = np.memmap(
    EMBEDDINGS_OUTPUT_FILENAME,
    dtype="float32",
    mode="w+",
    shape=(len(image_paths), EMBEDDING_DIM),
)

for i in tqdm(range(0, len(image_paths), BATCH_SIZE)):
    # Load batch of images
    batch_paths = image_paths[i : i + BATCH_SIZE]
    images = [preprocess(Image.open(p)).unsqueeze(0).to(device) for p in batch_paths]
    images = torch.cat(images)

    with torch.no_grad():
        image_features = model.encode_image(images).cpu().numpy()
        embeddings[i : i + BATCH_SIZE] = image_features

embeddings.flush()
