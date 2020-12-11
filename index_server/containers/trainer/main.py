import functools
import threading

import backoff
import numpy as np
import requests
from flask import Flask, request, abort

from typing import Callable, Dict, List

from interactive_index import InteractiveIndex
from interactive_index.config import auto_config

import config


# Step 1: Load saved embeddings into memory
def load(
    paths: List[str], sample_rate: float, reduction: Callable[[np.ndarray], np.ndarray]
):
    all_embeddings = []

    # Each file is a np.save'd Dict[int, np.ndarray] where each value is N x D
    for path in paths:
        embedding_dict = np.load(
            path, allow_pickle=True
        ).item()  # type: Dict[int, np.ndarray]

        for embeddings in embedding_dict.values():
            if sample_rate:
                n = embeddings.shape[0]
                n_sample = np.random.binomial(n, sample_rate)
                if n_sample == 0:
                    continue
                elif n_sample < n:
                    sample_inds = np.random.choice(n, n_sample)
                    embeddings = embeddings[sample_inds]
            all_embeddings.append(reduction(embeddings))

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


working_lock = threading.Lock()
app = Flask(__name__)


@app.route("/", methods=["POST"])
def start():
    try:
        payload = request.json or {}
        args = (
            list(payload["paths"]),
            float(payload.get("sample_rate", 1.0)),
            bool(payload.get("average")),
            int(payload["n_total"]),
            bool(payload.get("inner_product")),
            str(payload["job_id"]),
            str(payload["index_id"]),
            str(payload["url"]),
            working_lock,
        )
    except Exception as e:
        abort(400, description=str(e))

    if not working_lock.acquire(blocking=False):
        abort(503, description="Busy")

    thread = threading.Thread(
        target=main,
        args=args,
    )
    thread.start()
    return "Started"


# TODO(mihirg): Turn these into docstrings
# @click.command()
# @click.argument(
#     "paths",
#     type=click.Path(exists=True, dir_okay=False),
#     nargs=-1,
#     help="Paths to saved embedding dictionaries.",
# )
# @click.option(
#     "--sample_rate",
#     default=1.0,
#     type=click.FloatRange(0.0, 1.0),
#     help="Fraction of saved embeddings to randomly sample for training.",
# )
# @click.option(
#     "--average",
#     is_flag=True,
#     help="Average embeddings for each key in saved dictionary.",
# )
# @click.option(
#     "--n_total",
#     required=True,
#     type=click.IntRange(1),
#     help="Estimated total number of embeddings that will be added to index.",
# )
# @click.option(
#     "--inner_product",
#     is_flag=True,
#     help="Use inner product metric rather than L2 distance.",
# )
# @click.option("--job_id", required=True, help="Index build job identifier.")
# @click.option("--index_id", required=True, help="Unique index identifier within job.")
# @click.option("--url", required=True, help="Webhook to PUT to after completion.")
def main(
    paths, sample_rate, average, n_total, inner_product, job_id, index_id, url, lock
):
    try:
        reduction = (
            functools.partial(np.mean, axis=1, keepdims=True)
            if average
            else (lambda x: x)
        )
        embeddings = load(paths, sample_rate, reduction)
        index_dir = config.INDEX_DIR_TMPL.format(job_id, index_id)
        metric = "inner product" if inner_product else "L2"

        train(embeddings, n_total, metric, index_dir)
        notify(url, {"job_id": job_id, "index_id": index_id, "index_dir": index_dir})
    except Exception:
        raise
    finally:
        lock.release()
