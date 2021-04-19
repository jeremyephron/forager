from __future__ import annotations

import asyncio
from dataclasses import dataclass
from enum import IntEnum
import numpy as np
import os
from pathlib import Path
import time
import uuid
import logging
import aiohttp
import json

from typing import Any, Callable, Dict, List, Optional, Set
from interactive_index.config import auto_config
from knn import utils
from knn.utils import JSONType
from knn.jobs import MapReduceJob, MapperSpec
from knn.reducers import Reducer

from index_jobs import Trainer

import config

logger = logging.getLogger("index_server")


class BGSplitTrainingJob:
    def __init__(
        self,
        pos_paths: List[str],
        neg_paths: List[str],
        unlabeled_paths: List[str],
        aux_labels_path: str,
        model_name: str,
        model_id: str,
        resume_from: Optional[str],
        trainer: Trainer,
        cluster_mount_parent_dir: Path,
        session: aiohttp.ClientSession,
    ):
        self.model_name = model_name
        self.pos_paths = pos_paths
        self.neg_paths = neg_paths
        self.unlabeled_paths = unlabeled_paths
        self.aux_labels_path = aux_labels_path
        self.model_id = model_id
        self.resume_from = resume_from
        self.trainer = trainer
        self.cluster_mount_parent_dir = cluster_mount_parent_dir
        self.session = session

        self.started = False
        self.finished = asyncio.Event()
        self.failed = asyncio.Event()
        self.failure_reason: Optional[str] = None
        self.model_dir: Optional[str] = None
        self.profiling: Dict[str, float] = {}
        self.model_checkpoint: Optional[str] = None

        self._failed_or_finished = asyncio.Condition()

        # Will be initialized later
        self.model_kwargs: Dict[str, Any] = {}
        self._start_time: Optional[float] = None
        self._end_time: Optional[float] = None
        self._task: Optional[asyncio.Task] = None
        self._training_time_left: Optional[float] = None

    def configure_model(self):
        self.model_kwargs = dict(
            max_ram=config.BGSPLIT_TRAINING_MAX_RAM,
            aux_labels_path=self.aux_labels_path,
            resume_from=self.resume_from,
        )

    @property
    def status(self):
        end_time = self._end_time or time.time()
        start_time = self._start_time or end_time
        return {
            "started": self.started,
            "finished": self.finished.is_set(),
            "failed": self.failed.is_set(),
            "failure_reason": self.failure_reason,
            "elapsed_time": end_time - start_time,
            "training_time_left": self._training_time_left or -1,
            "profiling": self.profiling,
        }

    def start(self):
        self.started = True
        self._task = self.run_in_background()

    @utils.unasync_as_task
    async def run_in_background(self):
        self.configure_model()

        async with self.trainer as trainer_url:
            self._start_time = time.time()

            # TODO(mihirg): Add exponential backoff/better error handling
            try:
                request = self._construct_request()

                while not self.finished.is_set():
                    async with self._failed_or_finished:
                        async with self.session.post(
                            trainer_url, json=request
                        ) as response:
                            if response.status != 200:
                                await asyncio.sleep(5)
                                continue
                        await self._failed_or_finished.wait()
            except asyncio.CancelledError:
                pass
            finally:
                self._end_time = time.time()

    async def stop(self):
        if self._task is not None and not self._task.done():
            self._task.cancel()
            await self._task

    async def handle_result(self, result: JSONType):
        logger.debug(f"Train ({self.model_name}): recieved status update {result}")
        if result.get("success"):
            self.model_dir = result["model_dir"]
            self.model_checkpoint = result["model_checkpoint"]
            self.profiling = result["profiling"]
            self.finished.set()

        if result.get("failed"):
            self.failure_reason = result["reason"] if "reason" in result else None
            self.failed.set()
            self.finished.set()

        self._training_time_left = result.get("training_time_left", None)

        async with self._failed_or_finished:
            self._failed_or_finished.notify()

    @property
    def mounted_index_dir(self) -> Path:
        assert self.model_dir is not None
        return self.cluster_mount_parent_dir / self.model_dir.lstrip(os.sep)

    def _construct_request(self) -> JSONType:
        return {
            "train_positive_paths": self.pos_paths,
            "train_negative_paths": self.neg_paths,
            "train_unlabeled_paths": self.unlabeled_paths,
            "val_positive_paths": [],
            "val_negative_paths": [],
            "val_unlabeled_paths": [],
            "model_kwargs": self.model_kwargs,
            "model_id": self.model_id,
            "model_name": self.model_name,
            "notify_url": config.BGSPLIT_TRAINER_STATUS_CALLBACK,
        }


class BGSplitInferenceReducer(Reducer):
    @dataclass
    class Result:
        data_dir: Optional[str]

    CallbackType = Callable[[Result], None]

    def __init__(self, model_id: str, shared_dir: str, num_images: int):
        self.model_id = model_id
        self.shared_dir = shared_dir

        self.collected_data_dir = config.MODEL_OUTPUTS_PARENT_DIR / model_id
        os.makedirs(self.collected_data_dir, exist_ok=True)

        self.embeddings = np.memmap(
            self.collected_data_dir / config.EMBEDDING_FILE_NAME,
            dtype=np.float32,
            mode="w+",
            shape=(num_images, config.BGSPLIT_EMBEDDING_DIM),
        )
        self.scores = np.zeros((num_images,), dtype=np.float32)

    def handle_chunk_result(self, chunk, chunk_output):
        chunk_data_path = chunk_output
        chunk_data_path = os.path.join(self.shared_dir, chunk_data_path[1:])
        # Read into memory
        data = np.load(chunk_data_path, allow_pickle=True).item()

        self.embeddings[data["ids"]] = data["embeddings"]
        self.scores[data["ids"]] = data["scores"]

    def handle_result(self, input, output):
        pass

    @property
    def result(self) -> BGSplitInferenceReducer.Result:
        return BGSplitInferenceReducer.Result(
            data_dir=str(self.collected_data_dir),
        )

    def finish(self):
        self.embeddings.flush()
        np.save(self.collected_data_dir / config.MODEL_SCORES_FILE_NAME, self.scores)


class BGSplitInferenceJob:
    def __init__(
        self,
        paths: List[str],
        bucket: str,
        model_id: str,
        model_checkpoint_path: str,
        cluster: Cluster,
        session: aiohttp.ClientSession,
    ):
        self.paths = paths
        self.bucket = bucket
        self.model_id = model_id
        self.model_checkpoint_path = model_checkpoint_path
        self.cluster = cluster
        self.cluster_shared_dir = cluster.mount_parent_dir
        self.session = session

        self.started = False
        self.finished = asyncio.Event()
        self.failed = asyncio.Event()
        self.failure_reason: Optional[str] = None
        self.embedding_dir: Optional[str] = None
        self.profiling: Dict[str, float] = {}

        self._failed_or_finished = asyncio.Condition()

        # Will be initialized later
        self.job_args: Dict[str, Any] = {}
        self._start_time: Optional[float] = None
        self._end_time: Optional[float] = None
        self._task: Optional[asyncio.Task] = None
        self._time_left: Optional[float] = None
        self.result: Optional[BGSplitInferenceReducer.Result] = None
        self.mapper_job: Optiona[MapperJob] = None

    def configure_args(self):
        self.job_args = dict(
            max_ram=config.BGSPLIT_TRAINING_MAX_RAM,
            checkpoint_path=self.model_checkpoint_path,
        )

    @property
    def status(self):
        if self.mapper_job:
            prog = self.mapper_job.progress
            perf = self.mapper_job.performance
            end_time = self._end_time or time.time()
            start_time = self._start_time or end_time
            total_processed = prog["n_processed"] + prog["n_skipped"]
            total_left = prog["n_total"] - total_processed
            if total_processed > 0:
                time_left = total_left * (prog["elapsed_time"] / total_processed)
            else:
                time_left = -1
            if prog["finished"]:
                self.finished.set()
            return {
                "started": self.started,
                "finished": self.finished.is_set(),
                "elapsed_time": end_time - start_time,
                "time_left": time_left,
                "progress": prog,
                "performance": perf,
                "model_id": self.model_id,
                "output_dir": self.result.data_dir if self.result else None,
            }
        else:
            return {
                "started": self.started,
                "finished": self.finished.is_set(),
                "elapsed_time": 0,
                "time_left": -1,
                "model_id": self.model_id,
                "output_dir": None,
            }

    def start(self):
        self.started = True
        self._task = self.run_in_background()

    @utils.unasync_as_task
    async def run_in_background(self):
        self._start_time = time.time()
        self._construct_mapper()
        logger.info(f"BGSplit Map: started with {len(self.paths)} images")
        await self.mapper_job.run_until_complete(
            [{"path": path, "id": idx} for idx, path in enumerate(self.paths)]
        )
        logger.info(f"BGSplit Map: finished with {len(self.paths)} images")
        self.result = self.mapper_job.result
        self._end_time = time.time()

    async def stop(self):
        if self._task is not None and not self._task.done():
            self._task.cancel()
            await self._task

    def _construct_mapper(self) -> MapReduceJob:
        nproc = self.cluster.output["bgsplit_mapper_nproc"]
        n_mappers = int(
            self.cluster.output["num_bgsplit_mappers"]
            * config.BGSPLIT_MAPPER_REQUEST_MULTIPLE(nproc)
        )
        chunk_size = config.BGSPLIT_MAPPER_CHUNK_SIZE(nproc)
        self.mapper_job = MapReduceJob(
            mapper=MapperSpec(
                url=self.cluster.output["bgsplit_mapper_url"],
                # url=config.BGSPLIT_MAPPER_CLOUD_RUN_URL,
                n_mappers=n_mappers,
            ),
            reducer=BGSplitInferenceReducer(
                model_id=self.model_id,
                shared_dir=self.cluster_shared_dir,
                num_images=len(self.paths),
            ),
            mapper_args={
                "input_bucket": self.bucket,
                "return_type": "save",
                "checkpoint_path": self.model_checkpoint_path,
            },
            session=self.session,
            n_retries=config.BGSPLIT_MAPPER_NUM_RETRIES,
            chunk_size=chunk_size,
            request_timeout=config.BGSPLIT_MAPPER_REQUEST_TIMEOUT,
        )
        return self.mapper_job