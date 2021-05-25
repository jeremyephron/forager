from scipy.spatial.distance import pdist, squareform
import numpy as np


def run(
    embeddings_filename: str,
    num_embeddings: int,
    embedding_dim: int,
    distance_matrix_filename: str,
):
    embeddings = np.memmap(
        embeddings_filename,
        dtype="float32",
        mode="r",
        shape=(num_embeddings, embedding_dim),
    )
    embeddings_in_memory = np.copy(embeddings)

    distances_in_memory = squareform(
        pdist(embeddings_in_memory).astype(embeddings_in_memory.dtype)
    )

    distances = np.memmap(
        distance_matrix_filename,
        dtype=distances_in_memory.dtype,
        mode="w+",
        shape=distances_in_memory.shape,
    )
    distances[:] = distances_in_memory[:]
    distances.flush()
