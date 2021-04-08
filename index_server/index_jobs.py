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

from typing import Any, Callable, Dict, List, Optional, Set

import aiohttp

from interactive_index.config import auto_config

from knn import utils
from knn.utils import JSONType
from knn.reducers import Reducer

import config

logger = logging.getLogger("index_server")

class MapperReducer(Reducer):
    @dataclass
    class Result:
        num_images: int
        num_embeddings: int
        output_path_tmpls: List[str]
        finished: bool

    CallbackType = Callable[[Result], None]

    @dataclass
    class NotificationRequest:
        callback: MapperReducer.CallbackType
        on_num_images: Optional[int] = None
        on_num_embeddings: Optional[int] = None

    def __init__(self, notifications: Optional[List[NotificationRequest]] = None):
        self.notifications = {uuid.uuid4(): notif for notif in (notifications or ())}

        self.num_images = 0
        self.num_embeddings = 0
        self.output_path_tmpls: List[str] = []

        self.state_changed = asyncio.Condition()
        self.finished = asyncio.Event()

    def add_notification_request(self, notif: NotificationRequest):
        self.notifications[uuid.uuid4()] = notif

    def handle_chunk_result(self, chunk, chunk_output):
        self.output_path_tmpls.append(chunk_output)
        self.wake_gen()

    def handle_result(self, input, output):
        self.num_images += 1
        self.num_embeddings += output

        notification_keys = []
        for k, notif in self.notifications.items():
            if (
                notif.on_num_images is not None
                and self.num_images >= notif.on_num_images
            ) or (
                notif.on_num_embeddings is not None
                and self.num_embeddings >= notif.on_num_embeddings
            ):
                notification_keys.append(k)
        if not notification_keys:
            return

        callback_data = self.get_result(copy=True)
        for k in notification_keys:
            notif = self.notifications.pop(k)
            notif.callback(callback_data)

    def finish(self):
        self.finished.set()
        self.wake_gen()

    @property
    def result(self) -> Result:
        return self.get_result()

    def get_result(self, copy=False) -> Result:
        return MapperReducer.Result(
            self.num_images,
            self.num_embeddings,
            list(self.output_path_tmpls) if copy else self.output_path_tmpls,
            self.finished.is_set(),
        )

    @utils.unasync_as_task
    async def wake_gen(self):
        async with self.state_changed:
            self.state_changed.notify_all()

    async def output_path_tmpl_gen(self):
        i = 0
        while True:
            if i < len(self.output_path_tmpls):
                yield self.output_path_tmpls[i]
                i += 1
            elif self.finished.is_set():
                break
            else:
                async with self.state_changed:
                    await self.state_changed.wait()


class AdderReducer(Reducer):
    def __init__(self):
        self.shard_patterns: Set[str] = set()

    def handle_chunk_result(self, chunk, chunk_output):
        self.shard_patterns.add(chunk_output)

    def handle_result(self, input, output):
        pass

    @property
    def result(self) -> Set[str]:
        return self.shard_patterns


class IndexType(IntEnum):
    FULL = 0
    FULL_DOT = 1
    SPATIAL = 2
    SPATIAL_DOT = 3


class Trainer:
    def __init__(self, url: str):
        self.url = url
        self.lock = asyncio.Lock()

    async def __aenter__(self) -> str:  # returns endpoint
        await self.lock.acquire()
        return self.url

    async def __aexit__(self, type, value, traceback):
        self.lock.release()


class TrainingJob:
    def __init__(
        self,
        index_type: IndexType,
        dataset_num_images: int,
        index_id: str,
        trainer: Trainer,
        cluster_mount_parent_dir: Path,
        session: aiohttp.ClientSession,
    ):
        self.index_name = index_type.name
        self.dataset_num_images = dataset_num_images
        self.index_id = index_id
        self.trainer = trainer
        self.cluster_mount_parent_dir = cluster_mount_parent_dir
        self.session = session

        self.average = index_type in (IndexType.FULL, IndexType.FULL_DOT)
        self.inner_product = index_type in (IndexType.FULL_DOT, IndexType.SPATIAL_DOT)

        self.started = False
        self.finished = asyncio.Event()
        self.index_dir: Optional[str] = None
        self.profiling: Dict[str, float] = {}

        self._failed_or_finished = asyncio.Condition()

        # Will be initialized later
        self.index_kwargs: Dict[str, Any] = {}
        self._start_time: Optional[float] = None
        self._end_time: Optional[float] = None
        self._task: Optional[asyncio.Task] = None

    def make_notification_request_to_start_training(
        self, mapper_result: MapperReducer.Result, callback: MapperReducer.CallbackType
    ) -> MapperReducer.NotificationRequest:
        self.configure_index(mapper_result)
        if self.average:
            on_num_images = (
                config.TRAINER_N_CENTROIDS_MULTIPLE * self.index_kwargs["n_centroids"]
            )
            return MapperReducer.NotificationRequest(
                callback, on_num_images=on_num_images
            )
        else:
            on_num_embeddings = int(
                config.TRAINER_N_CENTROIDS_MULTIPLE
                * self.index_kwargs["n_centroids"]
                / config.TRAINER_EMBEDDING_SAMPLE_RATE
            )
            return MapperReducer.NotificationRequest(
                callback, on_num_embeddings=on_num_embeddings
            )

    def configure_index(self, mapper_result: MapperReducer.Result):
        n_vecs = (
            mapper_result.num_images
            if mapper_result.finished
            else self.dataset_num_images
        )
        if not self.average:
            n_vecs = int(
                mapper_result.num_embeddings / mapper_result.num_images * n_vecs
            )
        self.model_kwargs = dict(
            max_ram=config.BGSPLIT_TRAINING_MAX_RAM,
        )
        self.index_kwargs.update(
            vectors_per_index=config.INDEX_SUBINDEX_SIZE,
            metric="inner product" if self.inner_product else "L2",
            multi_id=True,
        )
        if (
            config.BUILD_UNCOMPRESSED_FULL_IMAGE_INDEX
            and self.average
            and self.inner_product
        ):
            # Don't apply any compression on the flat dot product index so that we can
            # extract raw full image embeddings (to accelerate SVM training later on)
            self.index_kwargs.update(direct_map="Hashtable")
        else:
            self.index_kwargs.update(
                transform="PCAR",
                transform_args=[config.INDEX_PCA_DIM],
                encoding="SQ",
                encoding_args=[config.INDEX_SQ_BYTES],
            )

    @property
    def status(self):
        end_time = self._end_time or time.time()
        start_time = self._start_time or end_time
        return {
            "started": self.started,
            "finished": self.finished.is_set(),
            "config": self.index_kwargs,
            "elapsed_time": end_time - start_time,
            "profiling": self.profiling,
        }




    def start(self, mapper_result: MapperReducer.Result):
        self.started = True
        self._task = self.run_in_background(mapper_result)

    @utils.unasync_as_task
    async def run_in_background(self, mapper_result: MapperReducer.Result):
        self.configure_index(mapper_result)

        async with self.trainer as trainer_url:
            self._start_time = time.time()

            # TODO(mihirg): Add exponential backoff/better error handling
            try:
                request = self._construct_request(mapper_result.output_path_tmpls)

                while not self.finished.is_set():
                    async with self._failed_or_finished:
                        async with self.session.post(
                            trainer_url, json=request
                        ) as response:
                            if response.status != 200:
                                continue
                        await self._failed_or_finished.wait()
                    await asyncio.sleep(5)
            except asyncio.CancelledError:
                pass
            finally:
                self._end_time = time.time()

    async def stop(self):
        if self._task is not None and not self._task.done():
            self._task.cancel()
            await self._task

    async def handle_result(self, result: JSONType):
        if result.get("success"):
            self.index_dir = result["index_dir"]
            self.profiling = result["profiling"]
            self.finished.set()

        async with self._failed_or_finished:
            self._failed_or_finished.notify()

    @property
    def mounted_index_dir(self) -> Path:
        assert self.index_dir is not None
        return self.cluster_mount_parent_dir / self.index_dir.lstrip(os.sep)

    def _construct_request(self, path_tmpls: List[str]) -> JSONType:
        return {
            "path_tmpls": path_tmpls,
            "index_kwargs": self.index_kwargs,
            "index_id": self.index_id,
            "index_name": self.index_name,
            "url": config.TRAINER_STATUS_CALLBACK,
            "sample_rate": (
                1.0 if self.average else config.TRAINER_EMBEDDING_SAMPLE_RATE
            ),
            "reduction": "average" if self.average else None,
        }


class LocalFlatIndex:
    # TODO(mihirg): Don't store entire dataset in memory long-term

    INDEX_FILENAME = "embeddings.npy"
    DISTANCE_MATRIX_FILENAME = "distances.npy"

    @classmethod
    def load(cls, dir: Path, num_images: int):
        self = cls()
        self.index = np.memmap(
            dir / self.INDEX_FILENAME,
            dtype=np.float32,
            mode="r",
            shape=(num_images, config.EMBEDDING_DIM),
        )
        self.distance_matrix = np.memmap(
            dir / self.DISTANCE_MATRIX_FILENAME,
            dtype=np.float32,
            mode="r",
            shape=(num_images, num_images),
        )
        return self

    @classmethod
    def create(cls, dir: Path, num_images: int, cluster_mount_parent_dir: Path):
        self = cls()
        self.index = np.memmap(
            dir / self.INDEX_FILENAME,
            dtype=np.float32,
            mode="w+",
            shape=(num_images, config.EMBEDDING_DIM),
        )
        self.distance_matrix = np.memmap(
            dir / self.DISTANCE_MATRIX_FILENAME,
            dtype=np.float32,
            mode="w+",
            shape=(num_images, num_images),
        )
        self.cluster_mount_parent_dir = cluster_mount_parent_dir
        return self

    # Don't use this directly - use a @classmethod constructor
    def __init__(self):
        self.index: Optional[np.ndarray] = None
        self.distance_matrix: Optional[np.ndarray] = None
        self.cluster_mount_parent_dir: Optional[Path] = None

    def add_from_file(self, path_tmpl: str):
        # Each file is a np.save'd Dict[int, np.ndarray] where each value is 1 x D
        assert self.index is not None and self.cluster_mount_parent_dir
        embedding_dict = np.load(
            self.cluster_mount_parent_dir / path_tmpl.format("average").lstrip(os.sep),
            allow_pickle=True,
        ).item()  # type: Dict[int, np.ndarray]

        # Populate local index
        for id, embeddings in embedding_dict.items():
            assert len(embeddings) == 1
            self.index[int(id)] = embeddings[0]

    def build_distance_matrix(self):
        self._pdist(self.index, self.distance_matrix)

    def save(self, dir: Path):
        assert self.index is not None and self.distance_matrix is not None
        self.index.flush()
        self.distance_matrix.flush()

    @staticmethod
    def _pdist(M, out):
        # TODO(mihirg): Consider using cosine similarity instead
        # Alternative to scipy's pdist that respects dtype; modified from
        # http://www.xavierdupre.fr/app/mlprodict/helpsphinx/notebooks/onnx_pdist.html
        n = M.shape[0]
        buffer = np.empty((n - 1, M.shape[1]), dtype=M.dtype)  # TODO(mihirg): Eliminate
        a = np.empty(n, dtype=M.dtype)
        for i in range(1, n):
            np.subtract(M[:i], M[i], out=buffer[:i])  # broadcasted substraction
            np.square(buffer[:i], out=buffer[:i])
            np.sum(buffer[:i], axis=1, out=a[:i])
            np.sqrt(a[i], out=a[:i])
            out[:i, i] = a[:i]
            out[i, :i] = a[:i]


class BGSplitTrainingJob:
    def __init__(
        self,
        pos_paths: List[str],
        neg_paths: List[str],
        unlabeled_paths: List[str],
        aux_labels_path: str,
        model_id: str,
        trainer: Trainer,
        cluster_mount_parent_dir: Path,
        session: aiohttp.ClientSession,
    ):
        self.model_name = 'BGSPLIT'
        self.pos_paths = pos_paths
        self.neg_paths = neg_paths
        self.unlabeled_paths = unlabeled_paths
        self.aux_labels_path = aux_labels_path
        self.model_id = model_id
        self.trainer = trainer
        self.cluster_mount_parent_dir = cluster_mount_parent_dir
        self.session = session

        self.started = False
        self.finished = asyncio.Event()
        self.failed = asyncio.Event()
        self.failure_reason: Optional[str] = None
        self.model_dir: Optional[str] = None
        self.profiling: Dict[str, float] = {}

        self._failed_or_finished = asyncio.Condition()

        # Will be initialized later
        self.model_kwargs: Dict[str, Any] = {}
        self._start_time: Optional[float] = None
        self._end_time: Optional[float] = None
        self._task: Optional[asyncio.Task] = None

    def configure_model(self):
        self.model_kwargs = dict(
            max_ram=config.BGSPLIT_TRAINING_MAX_RAM,
            aux_labels_path=self.aux_labels_path,
        )

    @property
    def status(self):
        end_time = self._end_time or time.time()
        start_time = self._start_time or end_time
        return {
            "started": self.started,
            "finished": self.finished.is_set(),
            "failed": self.failed.is_set(),
            "failure_reasoon": self.failure_reason,
            "elapsed_time": end_time - start_time,
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
