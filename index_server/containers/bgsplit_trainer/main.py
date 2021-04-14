import functools
import threading
import backoff
import requests
import pickle
import time
import traceback
import logging
import os.path
import json
import sys
import types
import numpy as np
from flask import Flask, request, abort
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Any

import config
from training_loop import TrainingLoop
from util import download

logger = logging.getLogger("bgsplit")
logger.setLevel(logging.DEBUG)
handler = logging.StreamHandler(sys.stdout)
handler.setLevel(logging.DEBUG)
logger.addHandler(handler)



# Step 3: Call webhook to indicate completion
@backoff.on_exception(backoff.expo, requests.exceptions.RequestException)
def notify(url: str, payload: Dict[str, Any]):
    r = requests.put(url, data=json.dumps(payload))
    r.raise_for_status()


@dataclass
class TrainingJob:
    train_positive_paths: List[str]
    train_negative_paths: List[str]
    train_unlabeled_paths: List[str]
    val_positive_paths: List[str]
    val_negative_paths: List[str]
    val_unlabeled_paths: List[str]
    model_kwargs: Dict[str, Any]

    model_id: str
    model_name: str
    notify_url: str

    _lock: threading.Lock

    _done: bool = False
    _done_lock: threading.Lock = field(default_factory=threading.Lock)


    def start(self):
        thread = threading.Thread(target=self.run, daemon=True)
        thread.start()

    def notify_status(
            self, success: bool=False, failed: bool=False, **kwargs):
        data={
            "model_id": self.model_id,
            "model_name": self.model_name,
            "success": success,
            "failed": failed,
            **kwargs,
        }
        logger.debug(f'Sending notify: {data}')
        notify(self.notify_url, data)

    def finish(self, success: bool, failed: bool=False, **kwargs):
        with self._done_lock:
            if self._done:
                return
            self._done = True

        self.notify_status(success=success, failed=failed, **kwargs)
        self._lock.release()

    @property
    def done(self):
        with self._done_lock:
            return self._done

    def run(self):
        # TODO(mihirg): Figure out how to handle errors like OOMs and CUDA errors,
        # maybe start a subprocess?
        try:
            start_time = time.perf_counter()

            aux_labels_path = self.model_kwargs['aux_labels_path']
            logger.info(f'Downloading aux labels: {aux_labels_path}')
            aux_labels = {}
            data = download(aux_labels_path)
            auxiliary_labels = pickle.loads(data)
            for p, v in auxiliary_labels.items():
                aux_labels[os.path.basename(p)] = v
            model_dir = config.MODEL_DIR_TMPL.format(
                self.model_id, self.model_name)
            self.model_kwargs['aux_labels'] = auxiliary_labels
            self.model_kwargs['model_dir'] = model_dir

            end_time = time.perf_counter()
            train_start_time = time.perf_counter()
            # Train
            logger.info('Creating training model')
            loop = TrainingLoop(
                model_kwargs=self.model_kwargs,
                train_positive_paths=self.train_positive_paths,
                train_negative_paths=self.train_negative_paths,
                train_unlabeled_paths=self.train_unlabeled_paths,
                val_positive_paths=self.val_positive_paths,
                val_negative_paths=self.val_negative_paths,
                val_unlabeled_paths=self.val_unlabeled_paths,
                notify_callback=self.notify_status
            )
            logger.info('Running training')
            self.last_checkpoint_path = loop.run()
            end_time = time.perf_counter()
        except Exception as e:
            logger.exception(f'Exception: {traceback.print_exc()}')
            self.finish(False, failed=True, reason=str(e))
        else:
            logger.info('Finished training')
            self.finish(
                True,
                model_dir=model_dir,
                model_checkpoint=self.last_checkpoint_path,
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

    payload["_lock"] = working_lock
    logger.debug(f'Received job payload: {payload}')
    current_job = TrainingJob(**payload)
    current_job.start()
    logger.debug(f'Started job: {payload}')
    return "Started"
