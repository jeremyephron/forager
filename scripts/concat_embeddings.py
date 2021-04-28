from pathlib import Path
import json

import numpy as np

EXISTING_IDENTIFIERS_FILENAME = "../../identifiers.json"
ADDITIONAL_IDENTIFIERS_FILENAME = "../../val_identifiers.json"

EXISTING_EMBEDDINGS_FILENAME = "embeddings.npy"
ADDITIONAL_EMBEDDINGS_FILENAME = "../../waymo-val-r50/embeddings_res5.npy"

NEW_EMBEDDINGS_FILENAME = "embeddings_concat.npy"

EMBEDDING_DIM = 2048

num_existing = len(json.load(Path(EXISTING_IDENTIFIERS_FILENAME).open()))
num_additional = len(json.load(Path(ADDITIONAL_IDENTIFIERS_FILENAME).open()))

existing_embeddings = np.memmap(
    EXISTING_EMBEDDINGS_FILENAME,
    dtype="float32",
    mode="r",
    shape=(num_existing, EMBEDDING_DIM),
)
additional_embeddings = np.memmap(
    ADDITIONAL_EMBEDDINGS_FILENAME,
    dtype="float32",
    mode="r",
    shape=(num_additional, EMBEDDING_DIM),
)

new_embeddings = np.memmap(
    NEW_EMBEDDINGS_FILENAME,
    dtype="float32",
    mode="w+",
    shape=(num_existing + num_additional, EMBEDDING_DIM),
)
new_embeddings[:num_existing] = existing_embeddings[:]
new_embeddings[num_existing:] = additional_embeddings[:]
new_embeddings.flush()
