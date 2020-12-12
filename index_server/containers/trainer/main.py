from dataclasses import dataclass, field
import functools
import threading

import backoff
import numpy as np
import requests
from flask import Flask, request, abort

from typing import Callable, Dict, List, Optional

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


@dataclass
class TrainingJob:
    paths: List[str]  # Paths to saved embedding dictionaries
    n_total: int  # Estimated total number of embeddings that will be added to index

    job_id: str  # Index build job identifier
    index_id: str  # Unique index identifier within job
    url: str  # Webhook to PUT to after completion

    sample_rate: float = (
        1.0  # Fraction of saved embeddings to randomly sample for training
    )
    average: bool = (
        False  # Whether to average embeddings for each key in saved dictionary
    )
    inner_product: bool = (
        False  # Whether to use inner product metric rather than L2 distance
    )

    _done: bool = False
    _done_lock: threading.Lock = field(default_factory=threading.Lock)

    def start(self):
        thread = threading.Thread(target=self.run, daemon=True)
        thread.start()

    def stop(self):
        with self._done_lock:
            if self._done:
                return
            self._done = True

        notify(
            self.url,
            {"job_id": self.job_id, "index_id": self.index_id, "success": False},
        )

    @property
    def done(self):
        with self._done_lock:
            return self._done

    def run(self):
        reduction = (
            functools.partial(np.mean, axis=1, keepdims=True)
            if self.average
            else (lambda x: x)
        )
        embeddings = load(self.paths, self.sample_rate, reduction)
        index_dir = config.INDEX_DIR_TMPL.format(self.job_id, self.index_id)
        metric = "inner product" if self.inner_product else "L2"

        train(embeddings, self.n_total, metric, index_dir)
        with self._done_lock:
            if self._done:
                return
            self._done = True

        notify(
            self.url,
            {
                "job_id": self.job_id,
                "index_id": self.index_id,
                "success": True,
                "index_dir": index_dir,
            },
        )


current_job: Optional[TrainingJob] = None
app = Flask(__name__)


@app.route("/", methods=["POST"])
def start():
    global current_job
    if current_job and not current_job.done:
        abort(503, description="Busy")

    payload = request.json or {}
    current_job = TrainingJob(**payload)
    return "Started"


@app.teardown_appcontext
def stop(_):
    if current_job:
        current_job.stop()
