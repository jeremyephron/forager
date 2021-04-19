from pathlib import Path
import json

import numpy as np

ORIGINAL_IMAGE_PATH_LIST_FILENAME = "images.json"
FINAL_IMAGE_IDENTIFIERS_LIST_FILENAME = "../../identifiers.json"
EMBEDDINGS_FILENAME = "embeddings.npy"
EMBEDDING_DIM = 512

image_paths = json.load(Path(ORIGINAL_IMAGE_PATH_LIST_FILENAME).open())
image_identifiers = [p[p.rfind("/") + 1 : p.rfind(".")] for p in image_paths]
identifiers_to_indices = json.load(Path(FINAL_IMAGE_IDENTIFIERS_LIST_FILENAME).open())

embeddings = np.memmap(
    EMBEDDINGS_FILENAME,
    dtype="float32",
    mode="r+",
    shape=(len(image_paths), EMBEDDING_DIM),
)

copy = np.copy(embeddings)

for i, identifier in enumerate(image_identifiers):
    embeddings[identifiers_to_indices[identifier]] = copy[i]

embeddings.flush()
