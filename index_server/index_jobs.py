from __future__ import annotations

import asyncio
from collections import defaultdict
from dataclasses import dataclass
from enum import IntEnum
import os
from pathlib import Path
import time
import uuid

from typing import Any, Callable, Dict, List, Optional, Set

import aiohttp
import numpy as np

from interactive_index.config import auto_config

from knn import utils
from knn.utils import JSONType
from knn.reducers import Reducer

import config


class MapperReducer(Reducer):
    @dataclass
    class Result:
        num_images: int
        num_embeddings: int
        output_paths: List[str]
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
        self.output_paths: List[str] = []

        self.wake_gen = asyncio.Condition()
        self.finished = asyncio.Event()

    def add_notification_request(self, notif: NotificationRequest):
        self.notifications[uuid.uuid4()] = notif

    @utils.unasync_as_task
    async def handle_chunk_result(self, chunk, chunk_output):
        self.output_paths.append(chunk_output)
        async with self.wake_gen:
            self.wake_gen.notify_all()

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

    @utils.unasync_as_task
    async def finish(self):
        print("Mapper finished")
        self.finished.set()
        async with self.wake_gen:
            self.wake_gen.notify_all()

    @property
    def result(self) -> Result:
        return self.get_result()

    def get_result(self, copy=False) -> Result:
        return MapperReducer.Result(
            self.num_images,
            self.num_embeddings,
            list(self.output_paths) if copy else self.output_paths,
            self.finished.is_set(),
        )

    async def output_paths_gen(self):
        i = 0
        while True:
            if i < len(self.output_paths):
                yield self.output_paths[i]
                i += 1
            elif self.finished.is_set():
                break
            else:
                async with self.wake_gen:
                    await self.wake_gen.wait()


class AdderReducer(Reducer):
    def __init__(self):
        self.num_added_by_index_type = defaultdict(int)
        self.shard_tmpls: Set[str] = set()

    def handle_chunk_result(self, chunk, chunk_output):
        self.shard_tmpls.add(chunk_output)

    def handle_result(self, input, output):
        for index_type, num_added in output.items():
            self.num_added_by_index_type[index_type] += num_added

    @property
    def result(self) -> Set[str]:
        return self.shard_tmpls


def extract_pooled_embedding_from_mapper_output(output):
    return utils.base64_to_numpy(output).mean(axis=0).astype(np.float32)


class IndexType(IntEnum):
    FULL = 0
    SPATIAL = 1
    FULL_DOT = 2
    SPATIAL_DOT = 3


class TrainingJob:
    def __init__(
        self,
        index_type: IndexType,
        dataset_num_images: int,
        index_id: str,
        trainer_url: str,
        cluster_mount_parent_dir: Path,
        session: aiohttp.ClientSession,
    ):
        self.index_name = index_type.name
        self.dataset_num_images = dataset_num_images
        self.index_id = index_id
        self.trainer_url = trainer_url
        self.cluster_mount_parent_dir = cluster_mount_parent_dir
        self.session = session

        self.average = index_type in (IndexType.FULL, IndexType.FULL_DOT)
        self.inner_product = index_type in (IndexType.FULL_DOT, IndexType.SPATIAL_DOT)

        self.started = False
        self.finished = asyncio.Event()
        self.index_dir: Optional[str] = None

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
        num_images = (
            mapper_result.num_images
            if mapper_result.finished
            else self.dataset_num_images
        )
        if self.average:
            n_vecs = num_images
        else:
            n_vecs = int(
                mapper_result.num_embeddings / mapper_result.num_images * num_images
            )
        self.index_kwargs = auto_config(
            d=config.EMBEDDING_DIM,
            n_vecs=n_vecs,
            max_ram=config.TRAINING_MAX_RAM,
            pca_d=config.INDEX_PCA_DIM,
            sq=config.INDEX_SQ_BYTES,
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
        }

    def start(self, mapper_result: MapperReducer.Result):
        self.started = True
        self._task = asyncio.create_task(self.run_until_complete(mapper_result))

    async def run_until_complete(self, mapper_result: MapperReducer.Result):
        self._start_time = time.time()
        self.configure_index(mapper_result)

        # TODO(mihirg): Add exponential backoff/better error handling
        try:
            request = self._construct_request(mapper_result.output_paths)

            while not self.finished.is_set():
                async with self._failed_or_finished:
                    async with self.session.post(
                        self.trainer_url, json=request
                    ) as response:
                        if response.status != 200:
                            continue
                    await self._failed_or_finished.wait()
        finally:
            self._end_time = time.time()

    async def stop(self):
        if self._task is not None and not self._task.done():
            self._task.cancel()
            await self._task

    async def handle_result(self, result: JSONType):
        if result.get("success"):
            self.index_dir = result["index_dir"]
            self.finished.set()

        async with self._failed_or_finished:
            self._failed_or_finished.notify()

    @property
    def mounted_index_dir(self) -> Path:
        assert self.index_dir is not None
        return self.cluster_mount_parent_dir / self.index_dir.lstrip(os.sep)

    def _construct_request(self, paths: List[str]) -> JSONType:
        return {
            "paths": paths,
            "index_kwargs": self.index_kwargs,
            "index_id": self.index_id,
            "index_name": self.index_name,
            "url": config.TRAINER_STATUS_CALLBACK,
            "sample_rate": (
                1.0 if self.average else config.TRAINER_EMBEDDING_SAMPLE_RATE
            ),
            "average": self.average,
            "inner_product": self.inner_product,
        }
