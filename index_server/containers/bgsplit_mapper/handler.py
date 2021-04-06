import functools
import threading

import backoff
import numpy as np
import requests
import pickle
import time
import traceback
import os.path
from flask import Flask, request, abort
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Any

import config
from training_loop import TrainingLoop


# Step 3: Call webhook to indicate completion
@backoff.on_exception(backoff.expo, requests.exceptions.RequestException)
def notify(url: str, payload: Dict[str, str]):
    r = requests.put(url, data=payload)
    r.raise_for_status()


@dataclass
class InferenceJob:
    paths: List[str]
    model_kwargs: Dict[str, Any]

    checkpoint_path: str
    model_id: str
    model_name: str
    notify_url: str

    _done: bool = False
    _done_lock: threading.Lock = field(default_factory=threading.Lock)

    def start(self):
        thread = threading.Thread(target=self.run, daemon=True)
        thread.start()

    def finish(self, success: bool, **kwargs):
        with self._done_lock:
            if self._done:
                return
            self._done = True

        notify(
            self.notify_url,
            {
                "model_id": self.model_id,
                "model_name": self.model_name,
                "success": success,
                **kwargs,
            },
        )

    @property
    def done(self):
        with self._done_lock:
            return self._done

    def run(self):
        # TODO(mihirg): Figure out how to handle errors like OOMs and CUDA errors,
        # maybe start a subprocess?
        try:
            start_time = time.perf_counter()

            self.model_kwargs['aux_labels'] = auxiliary_labels
            self.model_kwargs['model_dir'] = model_dir

            end_time = time.perf_counter()
            train_start_time = time.perf_counter()
            # Train
            loop = TrainingLoop(
                model_kwargs=self.model_kwargs,
                train_positive_paths=self.train_positive_paths,
                train_negative_paths=self.train_negative_paths,
                train_unlabeled_paths=self.train_unlabeled_paths,
                val_positive_paths=self.val_positive_paths,
                val_negative_paths=self.val_negative_paths,
                val_unlabeled_paths=self.val_unlabeled_paths,
            )
            loop.run()
            end_time = time.perf_counter()
        except Exception as e:
            traceback.print_exc()
            self.finish(False, reason=str(e))
        else:
            self.finish(
                True,
                model_dir=model_dir,
                profiling=dict(
                    load_time=train_start_time - start_time,
                    train_time=end_time - train_start_time,
                ),
            )


working_lock = threading.Lock()
app = Flask(__name__)


@app.route("/", methods=["POST"])
def start():
    try:
        payload = request.json or {}
    except Exception as e:
        abort(400, description=str(e))

    if not working_lock.acquire(blocking=False):
        abort(503, description="Busy")

    payload["lock"] = working_lock
    current_job = TrainingJob(**payload)
    current_job.start()
    return "Started"
