import concurrent.futures

from typing import List

import numpy as np
import torch
import clip
from PIL import Image
from tqdm import tqdm

BATCH_SIZE = 1
EMBEDDING_DIM = 512

device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)
load_image = lambda path: preprocess(Image.open(path)).unsqueeze(0).to(device)


def run(image_paths: List[str], embeddings_output_filename: str):
    embeddings = np.memmap(
        embeddings_output_filename,
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
