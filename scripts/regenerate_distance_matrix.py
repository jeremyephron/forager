from pathlib import Path
import json

import numpy as np
from tqdm import tqdm

IMAGE_PATHS_FILENAME = "../../labels.json"
EMBEDDINGS_FILENAME = "embeddings.npy"
DISTANCE_MATRIX_FILENAME = "distances.npy"
EMBEDDING_DIM = 2048


def pdist(M, out):
    # TODO(mihirg): Consider using cosine similarity instead
    # Alternative to scipy's pdist that respects dtype; modified from
    # http://www.xavierdupre.fr/app/mlprodict/helpsphinx/notebooks/onnx_pdist.html
    n = M.shape[0]
    buffer = np.empty((n - 1, M.shape[1]), dtype=M.dtype)  # TODO(mihirg): Eliminate
    a = np.empty(n, dtype=M.dtype)
    for i in tqdm(range(1, n)):
        np.subtract(M[:i], M[i], out=buffer[:i])  # broadcasted substraction
        np.square(buffer[:i], out=buffer[:i])
        np.sum(buffer[:i], axis=1, out=a[:i])
        np.sqrt(np.max(a[i], 0), out=a[:i])
        out[:i, i] = a[:i]
        out[i, :i] = a[:i]


image_paths = json.load(Path(IMAGE_PATHS_FILENAME).open())
embeddings = np.memmap(
    EMBEDDINGS_FILENAME,
    dtype="float32",
    mode="r",
    shape=(len(image_paths), EMBEDDING_DIM),
)

embeddings_in_memory = np.copy(embeddings)
distances_in_memory = np.zeros((len(image_paths), len(image_paths)), dtype=np.float32)
pdist(embeddings_in_memory, distances_in_memory)

distances = np.memmap(
    DISTANCE_MATRIX_FILENAME,
    dtype=distances_in_memory.dtype,
    mode="w+",
    shape=distances_in_memory.shape,
)
distances[:] = distances_in_memory[:]
distances.flush()
