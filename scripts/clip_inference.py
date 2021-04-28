import concurrent.futures
import json
from pathlib import Path

import numpy as np
import torch
import clip
from PIL import Image
from tqdm import tqdm

BATCH_SIZE = 256
IMAGE_DIR = Path("/home/mihir/waymo/val")
EMBEDDING_DIM = 512

IMAGE_LIST_OUTPUT_FILENAME = "images.json"
EMBEDDINGS_OUTPUT_FILENAME = "embeddings.npy"

device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)
load_image = lambda path: preprocess(Image.open(path)).unsqueeze(0).to(device)

image_paths = list(IMAGE_DIR.glob("*.jpeg"))
json.dump([str(p) for p in image_paths], Path(IMAGE_LIST_OUTPUT_FILENAME).open("w"))

embeddings = np.memmap(
    EMBEDDINGS_OUTPUT_FILENAME,
    dtype="float32",
    mode="w+",
    shape=(len(image_paths), EMBEDDING_DIM),
)

with concurrent.futures.ThreadPoolExecutor() as executor:
    for i in tqdm(range(0, len(image_paths), BATCH_SIZE)):
        # Load batch of images
        batch_paths = image_paths[i : i + BATCH_SIZE]
        images = torch.cat(list(executor.map(load_image, batch_paths)))

        with torch.no_grad():
            image_features = model.encode_image(images)
            image_features /= image_features.norm(dim=-1, keepdim=True)
            embeddings[i : i + BATCH_SIZE] = image_features.cpu().numpy()

embeddings.flush()
