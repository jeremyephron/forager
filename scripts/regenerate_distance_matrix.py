from pathlib import Path
import json

from scipy.spatial.distance import pdist, squareform
import numpy as np

IMAGE_PATHS_FILENAME = "../../labels.json"
EMBEDDINGS_FILENAME = "embeddings.npy"
DISTANCE_MATRIX_FILENAME = "distances.npy"
EMBEDDING_DIM = 2048

image_paths = json.load(Path(IMAGE_PATHS_FILENAME).open())
embeddings = np.memmap(
    EMBEDDINGS_FILENAME,
    dtype="float32",
    mode="r",
    shape=(len(image_paths), EMBEDDING_DIM),
)

embeddings_in_memory = np.copy(embeddings)
distances_in_memory = squareform(
    pdist(embeddings_in_memory).astype(embeddings_in_memory.dtype)
)

distances = np.memmap(
    DISTANCE_MATRIX_FILENAME,
    dtype=distances_in_memory.dtype,
    mode="w+",
    shape=distances_in_memory.shape,
)
distances[:] = distances_in_memory[:]
distances.flush()
