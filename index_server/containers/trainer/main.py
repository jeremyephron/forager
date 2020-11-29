import pickle
import random

import backoff
import click
import numpy as np
import requests

from typing import Dict, List

from interactive_index import InteractiveIndex
from interactive_index.config import auto_config

from . import config


# Step 1: Load picked embeddings into memory
def load(paths: List[str], sample_rate: float):
    all_embeddings = []

    # Each file is a pickled Dict[str, np.ndarray] where each key is N x D
    for path in paths:
        with open(path, "rb") as f:
            embedding_dict = pickle.load(f)  # type: Dict[str, np.ndarray]

        for embeddings in embedding_dict.values():
            if sample_rate:
                n = embeddings.shape[0]
                n_sample = np.random.binomial(n, sample_rate)
                if n_sample == 0:
                    continue
                elif n_sample < n:
                    sample_inds = random.sample(range(n), n_sample)
                    embeddings = embeddings[sample_inds]

            all_embeddings.append(embeddings)

    return np.concatenate(all_embeddings)


# Step 2: Train index
def train(embeddings: np.ndarray, n_total: int, metric: str, index_dir: str):
    n, d = embeddings.shape

    index_kwargs = auto_config(
        d=d,
        n_vecs=n_total,
        max_ram=config.INDEX_MAX_RAM_BYTES,
        pca_d=config.INDEX_PCA_DIM,
        sq=config.INDEX_SQ_BYTES,
    )
    index_kwargs.update(
        tempdir=index_dir,
        vectors_per_index=config.INDEX_SUBINDEX_SIZE,
        use_gpu=config.INDEX_USE_GPU,
        train_on_gpu=config.INDEX_TRAIN_ON_GPU,
        metric=metric,
    )

    index = InteractiveIndex(**index_kwargs)
    index.train(embeddings)


# Step 3: Call webhook to indicate completion
@backoff.on_exception(backoff.expo, requests.exceptions.RequestException)
def notify(url: str, payload: Dict[str, str]):
    r = requests.put(url, data=payload)
    r.raise_for_status()


@click.command()
@click.argument(
    "paths",
    type=click.Path(exists=True, dir_okay=False),
    nargs=-1,
    help="Paths to pickled embedding dictionaries.",
)
@click.option(
    "--sample_rate",
    default=1.0,
    type=click.FloatRange(0.0, 1.0),
    help="Fraction of saved embeddings to randomly sample for training.",
)
@click.option(
    "--n_total",
    required=True,
    type=click.IntRange(1),
    help="Estimated total number of embeddings that will be added to index.",
)
@click.option(
    "--inner_product",
    is_flag=True,
    help="Use inner product metric rather than L2 distance.",
)
@click.option("--index_id", required=True, help="Unique index identifier.")
@click.option("--url", required=True, help="Webhook to PUT to after completion.")
def main(paths, sample_rate, n_total, inner_product, index_id, url):
    embeddings = load(paths, sample_rate)
    index_dir = config.INDEX_DIR_PATTERN.format(index_id)
    metric = "inner product" if inner_product else "L2"

    train(embeddings, n_total, metric, index_dir)
    notify(url, {"index_id": index_id, "index_dir": index_dir})


if __name__ == "__main__":
    main(auto_envvar_prefix=config.ENVVAR_PREFIX)
