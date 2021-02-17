import asyncio
import concurrent
from collections import defaultdict
from dataclasses import dataclass
import functools
import heapq
import itertools
import json
import logging
import operator
import os
from pathlib import Path
import shutil
import time
import uuid

import aiohttp
from bidict import bidict
from dataclasses_json import dataclass_json
import numpy as np
from sanic import Sanic
import sanic.response as resp
from sklearn import svm
from sklearn.metrics import accuracy_score

from typing import Callable, DefaultDict, Dict, List, Optional, Set, Tuple

from interactive_index import InteractiveIndex
from interactive_index.utils import sample_farthest_vectors

from knn import utils
from knn.clusters import TerraformModule
from knn.jobs import MapReduceJob, MapperSpec
from knn.reducers import Reducer, IsFinishedReducer, VectorReducer
from knn.utils import JSONType

import config
from index_jobs import (
    AdderReducer,
    IndexType,
    MapperReducer,
    Trainer,
    TrainingJob,
    LocalFlatIndex,
)
from utils import CleanupDict


# Create a logger for the server
logger = logging.getLogger("index_server")
logger.setLevel(logging.DEBUG)

# Create a file handler for the log
log_fh = logging.FileHandler("index_server.log")
log_fh.setLevel(logging.DEBUG)

# Create a console handler to print errors to console
log_ch = logging.StreamHandler()
log_ch.setLevel(logging.DEBUG)

# create formatter and add it to the handlers
formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
log_fh.setFormatter(formatter)
log_ch.setFormatter(formatter)

# Attach handlers
logger.addHandler(log_fh)
logger.addHandler(log_ch)


class LabeledIndex:
    LABELS_FILENAME = "labels.json"
    IDENTIFIERS_FILENAME = "identifiers.json"

    @dataclass_json
    @dataclass
    class QueryResult:
        id: int
        dist: float = 0.0
        spatial_dists: Optional[List[Tuple[int, float]]] = None
        label: str = ""

    @dataclass
    class FurthestQueryResult:
        id: int
        spatial_locs: Optional[List[int]] = None
        label: str = ""

    # Don't use this directly - use a @classmethod constructor
    def __init__(self, index_id: str, *args, **kwargs):
        self.index_id = index_id
        self.index_dir = config.INDEX_PARENT_DIR / self.index_id
        self.logger = logging.getLogger(
            f"index_server.LabeledIndex({self.index_id[:6]})"
        )
        self.ready = asyncio.Event()

        # Will be filled by each individual constructor
        # TODO(mihirg): Deprecate self.labels in favor of self.identifiers
        self.labels: List[str] = []
        self.identifiers: Optional[bidict[str, int]] = None
        self.indexes: Dict[IndexType, InteractiveIndex] = {}
        self.local_flat_index: Optional[LocalFlatIndex] = None

        # Will only be used by the start_building() pathway
        self.bucket: Optional[str] = None
        self.cluster: Optional[TerraformModule] = None
        self.http_session: Optional[aiohttp.ClientSession] = None
        self.mapper_job: Optional[MapReduceJob] = None
        self.build_local_flat_index_task: Optional[asyncio.Task] = None
        self.training_jobs: CleanupDict[IndexType, TrainingJob] = CleanupDict(
            lambda job: job.stop()
        )
        self.start_adding_eventually_task: Optional[asyncio.Task] = None
        self.adder_job: Optional[MapReduceJob] = None
        self.resizer_job: Optional[MapReduceJob] = None
        self.merge_task: Optional[asyncio.Task] = None
        self.cluster_unlock_fn: Optional[Callable[[None], None]] = None

    def query(
        self,
        query_vector: np.ndarray,
        num_results: int,
        num_probes: Optional[int] = None,
        use_full_image: bool = False,
        svm: bool = False,
    ) -> List[QueryResult]:
        assert self.ready.is_set()

        self.logger.info(f"Query: use_full_image={use_full_image}, svm={svm}")
        start = time.perf_counter()

        if use_full_image:
            index = self.indexes[IndexType.FULL_DOT if svm else IndexType.FULL]
            dists, (ids, _) = index.query(
                query_vector, num_results, n_probes=num_probes
            )
            assert len(ids) == 1 and len(dists) == 1
            sorted_results = [
                LabeledIndex.QueryResult(int(i), float(d))  # cast numpy types
                for i, d in zip(ids[0], dists[0])
                if i >= 0
            ]
        else:
            index = self.indexes[IndexType.SPATIAL_DOT if svm else IndexType.SPATIAL]
            dists, (ids, locs) = index.query(
                query_vector,
                config.QUERY_NUM_RESULTS_MULTIPLE * num_results,
                n_probes=num_probes,
            )
            assert len(ids) == 1 and len(locs) == 1 and len(dists) == 1

            # Gather lowest QUERY_PATCHES_PER_IMAGE distances for each image
            dists_by_id: DefaultDict[int, List[float]] = defaultdict(list)
            spatial_dists_by_id: DefaultDict[
                int, List[Tuple[int, float]]
            ] = defaultdict(list)
            for i, l, d in zip(ids[0], locs[0], dists[0]):
                i, l, d = int(i), int(l), float(d)  # cast numpy types
                if i >= 0 and len(dists_by_id[i]) < config.QUERY_PATCHES_PER_IMAGE:
                    dists_by_id[i].append(d)
                    spatial_dists_by_id[i].append((l, d))

            # Average them and resort
            result_gen = (
                LabeledIndex.QueryResult(i, sum(ds) / len(ds), spatial_dists_by_id[i])
                for i, ds in dists_by_id.items()
                if len(ds) == config.QUERY_PATCHES_PER_IMAGE
            )
            sorted_results = heapq.nsmallest(
                num_results, result_gen, operator.attrgetter("dist")
            )

        end = time.perf_counter()
        self.logger.debug(
            f"Query of size {query_vector.shape} with k={num_results}, "
            f"n_probes={num_probes}, n_centroids={index.n_centroids}, and "
            f"n_vectors={index.n_vectors} took {end - start:.3f}s, and "
            f"got {len(sorted_results)} results."
        )

        for result in sorted_results:
            result.label = self.labels[result.id]
        return sorted_results

    def query_farthest(
        self,
        query_vector: np.ndarray,
        fraction: float,
        max_samples: int,
        use_full_image: bool = False,
    ) -> List[FurthestQueryResult]:
        if use_full_image:
            ids, _ = sample_farthest_vectors(
                self.indexes[IndexType.FULL_DOT], query_vector, fraction, max_samples
            )
            results = [LabeledIndex.FurthestQueryResult(int(i)) for i in ids]
        else:
            ids, locs = sample_farthest_vectors(
                self.indexes[IndexType.SPATIAL_DOT],
                query_vector,
                fraction,
                config.QUERY_NUM_RESULTS_MULTIPLE * max_samples,
            )

            # Gather spatial locations for each image
            locs_by_id: DefaultDict[int, List[int]] = defaultdict(list)
            for i, l in zip(ids, locs):
                i, l = int(i), int(l)  # cast numpy type  # noqa: E741
                locs_by_id[i].append(l)

            # Return up to max_samples images with the highest number (but at least
            # QUERY_PATCHES_PER_IMAGE) of returned spatial locations
            result_gen = (
                LabeledIndex.FurthestQueryResult(i, locs)
                for i, locs in locs_by_id.items()
            )
            results = heapq.nlargest(
                max_samples, result_gen, lambda r: len(r.spatial_locs)
            )

        for result in results:
            result.label = self.labels[result.id]
        return results

    # CLEANUP

    def delete(self):
        assert self.ready.is_set()
        shutil.rmtree(self.index_dir)

    async def stop_building(self):
        # Map
        if self.mapper_job:
            await self.mapper_job.stop()
        if (
            self.build_local_flat_index_task
            and not self.build_local_flat_index_task.done()
        ):
            self.build_local_flat_index_task.cancel()
            await self.build_local_flat_index_task

        # Train
        await self.training_jobs.clear_async()

        # Add
        if (
            self.start_adding_eventually_task
            and not self.start_adding_eventually_task.done()
        ):
            self.start_adding_eventually_task.cancel()
            await self.start_adding_eventually_task
        if self.adder_job:
            await self.adder_job.stop()

        # Resize
        if self.resizer_job:
            await self.resizer_job.stop()

        # Close network connections
        if self.http_session:
            await self.http_session.close()

        # Unlock cluster
        if self.cluster_unlock_fn:
            self.cluster_unlock_fn()

        # Merge
        if self.merge_task:
            self.merge_task.cancel()
            await self.merge_task

        # Delete unnecessary intermediates from local disk
        if not self.ready.is_set():
            shutil.rmtree(self.index_dir)

    # INDEX CREATION

    @classmethod
    async def start_building(
        cls,
        cluster: TerraformModule,
        cluster_unlock_fn: Callable[[], None],
        bucket: str,
        paths: List[str],
        identifiers: List[str],
        *args,
        **kwargs,
    ):
        self = cls(str(uuid.uuid4()), *args, **kwargs)
        self.bucket = bucket
        self.http_session = utils.create_unlimited_aiohttp_session()

        # Randomly shuffle input images
        inds = np.arange(len(paths))
        np.random.shuffle(inds)
        self.labels = [paths[i] for i in inds]
        self.identifiers = bidict({identifiers[i]: i for i in inds})
        iterable = (
            {"id": i, "image": path, "augmentations": {}}
            for i, path in enumerate(self.labels)
        )

        # Wait for the cluster to start, then do some configuration for the Train step,
        # which will start automatically as soon as the Map step (below) has made
        # sufficient progress
        await asyncio.gather(cluster.ready.wait(), cluster.mounted.wait())
        self.cluster = cluster
        self.cluster_unlock_fn = cluster_unlock_fn

        trainers = [Trainer(url) for url in self.cluster.output["trainer_urls"]]
        for index_type in IndexType:
            self.training_jobs[index_type] = TrainingJob(
                index_type,
                len(paths),
                self.index_id,
                trainers[index_type.value % len(trainers)],
                self.cluster.mount_parent_dir,
                self.http_session,
            )

        # Step 1: "Map" input images to embedding files saved to shared disk
        # TODO(mihirg): Fail gracefully if entire Map, Train, Add, or Resize jobs fail
        notification_request_to_configure_indexes = MapperReducer.NotificationRequest(
            self.configure_indexes,
            on_num_images=config.NUM_IMAGES_TO_MAP_BEFORE_CONFIGURING_INDEX,
        )

        nproc = self.cluster.output["mapper_nproc"]
        n_mappers = int(
            self.cluster.output["num_mappers"] * config.MAPPER_REQUEST_MULTIPLE(nproc)
        )
        chunk_size = config.MAPPER_CHUNK_SIZE(nproc)
        self.mapper_job = MapReduceJob(
            MapperSpec(
                url=self.cluster.output["mapper_url"],
                n_mappers=n_mappers,
            ),
            MapperReducer([notification_request_to_configure_indexes]),
            {"input_bucket": self.bucket, "return_type": "save"},
            session=self.http_session,
            n_retries=config.MAPPER_NUM_RETRIES,
            chunk_size=chunk_size,
            request_timeout=config.MAPPER_REQUEST_TIMEOUT,
        )
        await self.mapper_job.start(iterable, self.start_training, len(paths))
        self.logger.info(f"Map: started with {len(paths)} images")

        # Start a background task that consumes Map outputs as they're generated and
        # builds a local flat index of full-image embeddings
        self.local_flat_index = LocalFlatIndex.create(
            len(self.labels), self.cluster.mount_parent_dir
        )
        self.build_local_flat_index_task = self.build_local_flat_index_in_background()

        # Start a background task that waits until the Train step (started automatically
        # per above) is done and then kicks off the Add step
        self.start_adding_eventually_task = self.start_adding_eventually()

        return self

    def configure_indexes(self, mapper_result: MapperReducer.Result):
        self.logger.debug(
            "Map: configuring indexes after successfully processing "
            f"{mapper_result.num_images} images"
        )

        # Once we've successfully computed embeddings for a few images, use the results
        # so far to configure the indexes (number of centroids, etc.), and then use that
        # index configuration to figure out the number of images/embeddings we need to
        # start training each index. We can't do this before starting the Map step
        # because, for the spatial indexes, we need to know how many spatial embeddings
        # there are per image (dependent on model and image resolution) in order to
        # estimate the total number of vectors that will be in the index, which informs
        # the index configuration.
        assert self.mapper_job
        for index_type, job in self.training_jobs.items():
            self.mapper_job.reducer.add_notification_request(
                job.make_notification_request_to_start_training(
                    mapper_result,
                    functools.partial(self.start_training, index_type=index_type),
                )
            )

    @utils.unasync_as_task
    async def build_local_flat_index_in_background(self):
        # Step 2: As the Map step runs, build a local flat index of the full-image
        # embeddings it generates to facilitate fast SVM queries and a distance matrix
        # to facilitate fast clustering
        with concurrent.futures.ThreadPoolExecutor(
            config.LOCAL_INDEX_BUILDING_NUM_THREADS
        ) as pool:
            coro_gen = (
                utils.run_in_executor(
                    self.local_flat_index.add_from_file, path_tmpl, executor=pool
                )
                async for path_tmpl in self.mapper_job.reducer.output_path_tmpl_gen()
            )
            async for task in utils.limited_as_completed_from_async_coro_gen(
                coro_gen, config.LOCAL_INDEX_BUILDING_NUM_THREADS
            ):
                await task

            self.logger.info("Local flat index: finished consuming Map output")
            await utils.run_in_executor(self.local_flat_index.build_distance_matrix)
            self.logger.info("Local flat index: finished building distance matrix")

    def start_training(
        self,
        mapper_result: MapperReducer.Result,
        index_type: Optional[IndexType] = None,
    ):
        if mapper_result.finished:
            self.logger.info(
                "Map: finished after successfully processing "
                f"{mapper_result.num_images} images"
            )

        # Step 3: "Train" each index once we have enough images/spatial embeddings (or
        # when the Map step finishes, in which case index_type=None indicating that we
        # should train all remaining indexes)
        index_types = [index_type] if index_type else iter(IndexType)
        for index_type in index_types:
            if self.training_jobs[index_type].started:
                continue
            self.training_jobs[index_type].start(mapper_result)
            self.logger.info(
                f"Train ({index_type.name}): started with "
                f"{mapper_result.num_embeddings} embeddings for "
                f"{mapper_result.num_images} images"
            )

    async def handle_training_status_update(self, result: JSONType):
        # Because training takes a long time, the trainer sends us back an HTTP request
        # on status changes rather than communicating over a single request/response.
        # This function is called by the Sanic endpoint and passes the status update
        # along to the relevant training job.
        index_type = IndexType[result["index_name"]]
        self.logger.debug(f"Train ({index_type.name}): recieved status update {result}")

        if index_type in self.training_jobs:
            await self.training_jobs[index_type].handle_result(result)

    @utils.unasync_as_task
    async def start_adding_eventually(self):
        self.index_dir.mkdir(parents=True, exist_ok=False)
        indexes = {}

        # Wait until all indexes are trained
        async for done in utils.as_completed_from_futures(
            [
                asyncio.create_task(job.finished.wait(), name=index_type.name)
                for index_type, job in self.training_jobs.items()
            ]
        ):
            assert isinstance(done, asyncio.Task)
            await done

            index_type = IndexType[done.get_name()]
            job = self.training_jobs[index_type]
            indexes[index_type.name] = {
                "reduction": "average" if job.average else None,
                "index_dir": job.index_dir,
            }
            self.logger.info(f"Train ({index_type.name}): finished")

            # Copy index training results to local disk before anything else gets
            # written into the index directory on the shared disk
            index_subdir = self.index_dir / index_type.name
            shutil.copytree(job.mounted_index_dir, index_subdir)
            self.logger.debug(
                f"Train ({index_type.name}): copied trained index from shared disk "
                f"({job.mounted_index_dir}) to local disk ({index_subdir})"
            )

        # TODO(mihirg): Fix skipping issue in Map and Add when started concurrently,
        # then remove this line!
        await self.mapper_job.reducer.finished.wait()

        # Step 4: As the Map step computes and saves embeddings, "Add" them into shards
        # of the newly trained indexes
        # TODO(mihirg): Consider adding to each index independently as training
        # finishes, then merging independently, then making indexes available on the
        # frontend for queries as they are completed
        nproc = self.cluster.output["adder_nproc"]
        n_mappers = int(
            self.cluster.output["num_adders"] * config.ADDER_REQUEST_MULTIPLE(nproc)
        )
        chunk_size = config.ADDER_CHUNK_SIZE(nproc)
        self.adder_job = MapReduceJob(
            MapperSpec(
                url=self.cluster.output["adder_url"],
                n_mappers=n_mappers,
            ),
            AdderReducer(),
            {"indexes": indexes},
            session=self.http_session,
            n_retries=config.ADDER_NUM_RETRIES,
            chunk_size=chunk_size,
            request_timeout=config.ADDER_REQUEST_TIMEOUT,
        )
        await self.adder_job.start(
            self.mapper_job.reducer.output_path_tmpl_gen(),
            self.start_resizing_and_merging,
        )  # iterable is an async generator that yields as the Map step produces outputs
        self.logger.info("Add: started")

    @utils.unasync_as_task
    async def start_resizing_and_merging(self, shard_patterns: Set[str]):
        self.logger.info(f"Add: finished with {len(shard_patterns)} shard patterns")
        self.merge_task = self.merge_indexes_in_background(shard_patterns)

        # Step 5: "Resize" images into small thumbnails so that the frontend can render
        # faster
        assert self.cluster  # just to silence type warnings
        iterable = (
            {"image": label, "identifier": identifier}
            for label, identifier in zip(self.labels, self.identifiers.keys())
        )
        nproc = self.cluster.output["resizer_nproc"]
        n_mappers = int(
            self.cluster.output["num_resizers"] * config.RESIZER_REQUEST_MULTIPLE(nproc)
        )
        chunk_size = config.RESIZER_CHUNK_SIZE(nproc)
        self.resizer_job = MapReduceJob(
            MapperSpec(
                url=self.cluster.output["resizer_url"],
                n_mappers=n_mappers,
            ),
            IsFinishedReducer(),
            {
                "input_bucket": self.bucket,
                "output_bucket": config.RESIZER_OUTPUT_BUCKET,
                "output_dir": config.RESIZER_OUTPUT_DIR_TMPL.format(self.index_id),
                "resize_max_height": config.RESIZER_MAX_HEIGHT,
            },
            session=self.http_session,
            n_retries=config.RESIZER_NUM_RETRIES,
            chunk_size=chunk_size,
            request_timeout=config.RESIZER_REQUEST_TIMEOUT,
        )
        await self.resizer_job.start(
            iterable,
            lambda _: self.logger.info("Resize: finished"),
        )
        self.logger.info("Resize: started")

    @utils.unasync_as_task
    async def merge_indexes_in_background(self, shard_patterns: Set[str]):
        loop = asyncio.get_running_loop()

        # Step 6: "Merge" shards from shared disk into final local index (in a thread
        # pool; because FAISS releases the GIL, this won't block the event loop)
        # TODO(mihirg): Consider deleting all unnecessary intermediates from NAS after
        self._load_local_indexes()
        with concurrent.futures.ThreadPoolExecutor(len(self.indexes)) as pool:
            futures = []
            for index_type, index in self.indexes.items():
                index_dir = self.training_jobs[index_type].mounted_index_dir
                future = asyncio.ensure_future(
                    loop.run_in_executor(
                        pool, self.merge_index, index, shard_patterns, index_dir
                    )
                )
                self.logger.info(f"Merge ({index_type.name}): started")
                future.add_done_callback(
                    lambda _, index_type=index_type: self.logger.info(
                        f"Merge ({index_type.name}): finished"
                    )
                )
                futures.append(future)

            await asyncio.gather(*futures)

        # Upload final index to Cloud Storage
        await self.build_local_flat_index_task
        await self.upload()

        # Wait for resizing to complete
        await self.resizer_job.reducer.finished.wait()

        self.logger.info("Finished building index")
        self.ready.set()
        self.cluster_unlock_fn()

    @staticmethod
    def merge_index(index: InteractiveIndex, shard_patterns: Set[str], index_dir: Path):
        shard_paths = [
            str(p.resolve())
            for shard_pattern in shard_patterns
            for p in index_dir.glob(shard_pattern)
        ]
        index.merge_partial_indexes(shard_paths)

    async def upload(self):
        # Dump labels, identifiers, full-image embeddings, and distance matrix
        json.dump(self.labels, (self.index_dir / self.LABELS_FILENAME).open("w"))
        json.dump(
            dict(self.identifiers),
            (self.index_dir / self.IDENTIFIERS_FILENAME).open("w"),
        )
        self.local_flat_index.save(self.index_dir)

        # Upload to Cloud Storage
        # TODO(mihirg): Speed up
        # https://medium.com/google-cloud/google-cloud-storage-large-object-upload-speeds-7339751eaa24
        proc = await asyncio.create_subprocess_exec(
            "gsutil",
            "-m",
            "cp",
            "-r",
            "-n",
            str(self.index_dir),
            config.INDEX_UPLOAD_GCS_PATH,
        )
        await proc.wait()

    @property
    def status(self):
        return {
            "map": self.mapper_job.status if self.mapper_job else {},
            "add": self.adder_job.status if self.adder_job else {},
            "train": {
                index_type.name: job.status
                for index_type, job in self.training_jobs.items()
            },
            "resize": self.resizer_job.status if self.resizer_job else {},
        }

    # INDEX LOADING

    @classmethod
    async def download(
        cls,
        index_id: str,
        *args,
        **kwargs,
    ) -> "LabeledIndex":
        self = cls(index_id, *args, **kwargs)

        # Download from Cloud Storage
        config.INDEX_PARENT_DIR.mkdir(parents=True, exist_ok=True)
        # TODO(mihirg): Speed up
        # https://medium.com/@duhroach/gcs-read-performance-of-large-files-bd53cfca4410
        proc = await asyncio.create_subprocess_exec(
            "gsutil",
            "-m",
            "cp",
            "-r",
            "-n",
            f"{config.INDEX_UPLOAD_GCS_PATH}{self.index_id}",
            str(config.INDEX_PARENT_DIR),
        )
        await proc.wait()

        # Initialize indexes
        self.labels = json.load((self.index_dir / self.LABELS_FILENAME).open())
        self.identifiers = bidict(
            json.load((self.index_dir / self.IDENTIFIERS_FILENAME).open())
        )
        self.local_flat_index = LocalFlatIndex.load(self.index_dir)
        self._load_local_indexes()
        self.logger.info(f"Finished loading index from {self.index_dir}")

        self.ready.set()
        return self

    def _load_local_indexes(self):
        self.indexes = {
            index_type: InteractiveIndex.load(str(self.index_dir / index_type.name))
            for index_type in IndexType
        }


# Start web server
app = Sanic(__name__)
app.update_config({"RESPONSE_TIMEOUT": config.SANIC_RESPONSE_TIMEOUT})


# CLUSTER


async def _start_cluster(cluster):
    # Create cluster
    # Hack(mihirg): just attach mounting-related attributes to the cluster object
    cluster.mounted = asyncio.Event()
    await cluster.apply()

    # Mount NFS
    cluster.mount_parent_dir = config.CLUSTER_MOUNT_DIR / cluster.id
    cluster.mount_parent_dir.mkdir(parents=True, exist_ok=False)

    cluster.mount_dir = cluster.mount_parent_dir / cluster.output[
        "nfs_mount_dir"
    ].lstrip(os.sep)
    cluster.mount_dir.mkdir()

    proc = await asyncio.create_subprocess_exec(
        "sudo",
        "mount",
        cluster.output["nfs_url"],
        str(cluster.mount_dir),
    )
    await proc.wait()
    cluster.mounted.set()


async def _stop_cluster(cluster):
    await asyncio.gather(cluster.ready.wait(), cluster.mounted.wait())

    # Unmount NFS
    proc = await asyncio.create_subprocess_exec(
        "sudo", "umount", "-f", "-l", str(cluster.mount_dir)
    )
    await proc.wait()
    try:
        shutil.rmtree(cluster.mount_parent_dir)
    except Exception:
        pass

    # Destroy cluster
    if not config.CLUSTER_REUSE_EXISTING:
        await cluster.destroy()


# TODO(mihirg): Automatically clean up inactive clusters
current_clusters: CleanupDict[str, TerraformModule] = CleanupDict(
    _stop_cluster, app.add_task, config.CLUSTER_CLEANUP_TIME
)


@app.route("/start_cluster", methods=["POST"])
async def start_cluster(request):
    cluster = TerraformModule(
        config.CLUSTER_TERRAFORM_MODULE_PATH, copy=not config.CLUSTER_REUSE_EXISTING
    )
    app.add_task(_start_cluster(cluster))
    cluster_id = cluster.id
    current_clusters[cluster_id] = cluster
    return resp.json({"cluster_id": cluster_id})


@app.route("/cluster_status", methods=["GET"])
async def cluster_status(request):
    cluster_id = request.args["cluster_id"][0]
    cluster = current_clusters.get(cluster_id)
    has_cluster = cluster is not None

    status = {
        "has_cluster": has_cluster,
        "ready": has_cluster and cluster.ready.is_set(),
    }
    return resp.json(status)


@app.route("/stop_cluster", methods=["POST"])
async def stop_cluster(request):
    cluster_id = request.json["cluster_id"]
    app.add_task(current_clusters.cleanup_key(cluster_id))
    return resp.text("", status=204)


# INDEX


current_indexes: CleanupDict[str, LabeledIndex] = CleanupDict(
    lambda job: job.stop_building()
)

# -> BUILD
# TODO(mihirg): Consider restructuring to simplfiy


@app.route("/start_job", methods=["POST"])
async def start_job(request):
    cluster_id = request.json["cluster_id"]
    bucket = request.json["bucket"]
    paths = request.json["paths"]
    identifiers = request.json["identifiers"]

    cluster = current_clusters[cluster_id]
    lock_id = current_clusters.lock(cluster_id)
    cluster_unlock_fn = functools.partial(current_clusters.unlock, cluster_id, lock_id)
    index = await LabeledIndex.start_building(
        cluster, cluster_unlock_fn, bucket, paths, identifiers
    )

    index_id = index.index_id
    current_indexes[index_id] = index
    return resp.json({"index_id": index_id})


@app.route(config.TRAINER_STATUS_ENDPOINT, methods=["PUT"])
async def training_status(request):
    index_id = request.json["index_id"]
    if index_id in current_indexes:
        await current_indexes[index_id].handle_training_status_update(request.json)
    return resp.text("", status=204)


@app.route("/job_status", methods=["GET"])
async def job_status(request):
    index_id = request.args["index_id"][0]
    if index_id in current_indexes:
        index = current_indexes[index_id]
        status = index.status
        status["has_index"] = index.ready.is_set()
    else:
        status = {"has_index": False}
    return resp.json(status)


# TODO(mihirg): Do we even need this function? It's not exposed on the frontend.
@app.route("/stop_job", methods=["POST"])
async def stop_job(request):
    index_id = request.json["index_id"]
    app.add_task(current_indexes.cleanup_key(index_id))
    return resp.text("", status=204)


# -> POST-BUILD


async def _download_index(index_id):
    if index_id not in current_indexes:
        current_indexes[index_id] = await LabeledIndex.download(index_id)


@app.route("/download_index", methods=["POST"])
async def download_index(request):
    index_id = request.json["index_id"]
    app.add_task(_download_index(index_id))
    return resp.text("", status=204)


# TODO(mihirg): Do we even need this function?
@app.route("/delete_index", methods=["POST"])
async def delete_index(request):
    index_id = request.json["index_id"]
    await current_indexes.pop(index_id).delete()
    return resp.text("", status=204)


# QUERY
# TODO(mihirg): Change input from form encoding to JSON
# TODO(all): Clean up this code


def extract_embedding_from_mapper_output(output: str) -> np.ndarray:
    return np.squeeze(utils.base64_to_numpy(output), axis=0)


class BestMapper:
    def __init__(self, cluster_id: str):
        self.cluster_id = cluster_id
        self.lock_id: Optional[str] = None

    async def __aenter__(self) -> MapperSpec:  # returns endpoint
        cluster = current_clusters.get(self.cluster_id)
        if cluster and cluster.ready.is_set():
            self.lock_id = current_clusters.lock(self.cluster_id)

            nproc = cluster.output["mapper_nproc"]
            n_mappers = int(
                cluster.output["num_mappers"] * config.MAPPER_REQUEST_MULTIPLE(nproc)
            )
            return MapperSpec(url=cluster.output["mapper_url"], n_mappers=n_mappers)
        else:
            return MapperSpec(
                url=config.MAPPER_CLOUD_RUN_URL, n_mappers=config.CLOUD_RUN_N_MAPPERS
            )

    async def __aexit__(self, type, value, traceback):
        if self.lock_id:
            current_clusters.unlock(self.cluster_id, self.lock_id)


@app.route("/query_index", methods=["POST"])
async def query_index(request):
    image_paths = request.json["paths"]
    cluster_id = request.json["cluster_id"]
    bucket = request.json["bucket"]
    patches = [
        [float(patch[k]) for k in ("x1", "y1", "x2", "y2")]
        for patch in request.json["patches"]
    ]  # [0, 1]^2
    index_id = request.json["index_id"]
    num_results = int(request.json["num_results"])
    augmentations = request.json.get("augmentations", [])

    augmentation_dict = {}
    for i in range(len(augmentations) // 2):
        augmentation_dict[augmentations[2 * i]] = float(augmentations[2 * i + 1])

    use_full_image = bool(request.json.get("use_full_image", False))

    # Generate query vector as average of patch embeddings
    async with BestMapper(cluster_id) as mapper:
        job = MapReduceJob(
            mapper,
            VectorReducer(
                VectorReducer.PoolingType.AVG,
                extract_func=extract_embedding_from_mapper_output,
            ),
            {"input_bucket": bucket, "reduction": "average"},
            n_retries=config.CLOUD_RUN_N_RETRIES,
            chunk_size=1,
        )
        query_vector = await job.run_until_complete(
            [
                {
                    "image": image_path,
                    "patch": patch,
                    "augmentations": augmentation_dict,
                }
                for image_path, patch in zip(image_paths, patches)
            ]
        )

    # Run query and return results
    query_results = current_indexes[index_id].query(
        query_vector, num_results, None, use_full_image, False
    )
    return resp.json({"results": [r.to_dict() for r in query_results]})


@app.route("/active_batch", methods=["POST"])
async def active_batch(request):
    image_paths = request.json["paths"]  # Paths of seed images (at first from google)
    bucket = request.json["bucket"]
    cluster_id = request.json["cluster_id"]
    index_id = request.json["index_id"]
    num_results = int(request.json["num_results"])
    augmentations = request.json.get("augmentations", [])

    augmentation_dict = {}
    for i in range(len(augmentations) // 2):
        augmentation_dict[augmentations[2 * i]] = float(augmentations[2 * i + 1])

    use_full_image = True

    # Generate query vector as average of patch embeddings
    async with BestMapper(cluster_id) as mapper:
        job = MapReduceJob(
            mapper,
            VectorReducer(extract_func=extract_embedding_from_mapper_output),
            {"input_bucket": bucket, "reduction": "average"},
            n_retries=config.CLOUD_RUN_N_RETRIES,
            chunk_size=1,
        )
        query_vectors = await job.run_until_complete(
            [
                {"image": image_path, "augmentations": augmentation_dict}
                for image_path in image_paths
            ]
        )

    all_results = {}
    # Results per vector
    perVector = (int)(num_results / len(query_vectors)) + 2
    for vec in query_vectors:
        # Return nearby images
        query_results = current_indexes[index_id].query(
            np.float32(vec),
            perVector,
            None,
            use_full_image,
            False,  # False bc just standard nearest neighbor
        )

        # Remove duplicates; for each image, include closest result
        for result in query_results:
            label = result.label
            if label not in all_results or all_results[label].dist > result.dist:
                all_results[label] = result

    return resp.json(
        {"results": [r.to_dict() for r in all_results.values()]}
    )  # unordered by distance for now


# TODO(mihirg): Reuse these instances (maybe in an ExpiringDict, storing embeddings in
# a Chest internally?) to cache embeddings and speed up repeatedly iterating on training
# an SVM on a particular dataset
class SVMExampleReducer(Reducer):
    def __init__(self):
        self.labels: List[int] = []
        self.embeddings: List[np.ndarray] = []

    def handle_result(self, input: JSONType, output: str):
        label = int(bool(input["label"]))
        self.labels.append(label)
        self.embeddings.append(extract_embedding_from_mapper_output(output))

    @property
    def result(self) -> Tuple[np.ndarray, np.ndarray]:  # features, labels
        return np.stack(self.embeddings), np.array(self.labels)


@app.route("/query_svm", methods=["POST"])
async def query_svm(request):
    index_id = request.json["index_id"]
    cluster_id = request.json["cluster_id"]
    bucket = request.json["bucket"]
    pos_image_paths = request.json["positive_paths"]
    pos_patches = [
        [float(patch[k]) for k in ("x1", "y1", "x2", "y2")]
        for patch in request.json["positive_patches"]
    ]  # [0, 1]^2
    neg_image_paths = request.json["negative_paths"]
    num_results = int(request.json["num_results"])
    mode = request.json["mode"]
    use_full_image = bool(request.json.get("use_full_image", False))

    # Automatically label `autolabel_max_vectors` vectors randomly sampled from the
    # bottom `autolabel_percent`% of the previous SVM's results as negative
    index = current_indexes[index_id]

    prev_svm_vector = utils.base64_to_numpy(request.json["prev_svm_vector"])
    autolabel_percent = float(request.json["autolabel_percent"])
    autolabel_max_vectors = int(request.json["autolabel_max_vectors"])
    log_id_string = request.json["log_id_string"]

    if (
        prev_svm_vector is not None
        and autolabel_percent > 0
        and autolabel_max_vectors > 0
    ):
        already_labeled_image_paths = set(
            itertools.chain(pos_image_paths, neg_image_paths)
        )
        autolabel_results = index.query_farthest(
            prev_svm_vector,
            autolabel_percent / 100,  # percentage to fraction!
            autolabel_max_vectors,
            use_full_image,
        )
        autolabel_image_paths = [
            r.label
            for r in autolabel_results
            if r.label not in already_labeled_image_paths
        ]
    else:
        autolabel_image_paths = []

    # Generate training vectors
    async with BestMapper(cluster_id) as mapper:
        job = MapReduceJob(
            mapper,
            SVMExampleReducer(),
            {"input_bucket": bucket, "reduction": "average"},
            n_retries=config.CLOUD_RUN_N_RETRIES,
            chunk_size=1,
        )
        pos_inputs = [
            {"image": image_path, "patch": patch, "label": 1}
            for image_path, patch in zip(pos_image_paths, pos_patches)
        ]
        neg_inputs = [
            {"image": image_path, "label": 0} for image_path in neg_image_paths
        ]
        auto_inputs = [
            {"image": image_path, "label": 0} for image_path in autolabel_image_paths
        ]

        logger.info(
            f"{log_id_string} - Starting SVM training vector computation: {len(pos_inputs)} positives, "
            f"{len(neg_inputs)} negatives, {len(auto_inputs)} auto-negatives"
        )
        training_features, training_labels = await job.run_until_complete(
            itertools.chain(pos_inputs, neg_inputs, auto_inputs)
        )
        logger.info(
            f"{log_id_string} - Finished SVM vector computation in {job.elapsed_time:.3f}s"
        )
        logger.debug(
            f"{log_id_string} - Vector computation performance: {job.performance}"
        )

    # Train SVM
    logger.debug(f"{log_id_string} - Starting SVM training")
    start_time = time.perf_counter()

    model = svm.SVC(kernel="linear")
    model.fit(training_features, training_labels)
    predicted = model.predict(training_features)

    end_time = time.perf_counter()
    logger.info(
        f"{log_id_string} - Finished training SVM in {end_time - start_time:.3f}s"
    )
    logger.debug(
        f"{log_id_string} - SVM accuracy: {accuracy_score(training_labels, predicted)}"
    )

    if mode == "svmPos" or mode == "spatialSvmPos":
        # Evaluate the SVM by querying index
        w = model.coef_  # This will be the query vector
        # Also consider returning the support vectors; good to look at examples along
        # hyperplane
        w = np.float32(w[0] * 1000)

        augmentations = request.json.get("augmentations", [])

        augmentation_dict = {}
        for i in range(len(augmentations) // 2):
            augmentation_dict[augmentations[2 * i]] = float(augmentations[2 * i + 1])

        # Run query and return results
        query_results = index.query(w, num_results, None, use_full_image, True)
        return resp.json(
            {
                "results": [r.to_dict() for r in query_results],
                "svm_vector": utils.numpy_to_base64(w),
            }
        )
    elif mode == "svmBoundary":
        # Get samples close to boundary vectors
        # For now, looks like most vectors end up being support vectors since
        # underparamtrized system
        sv = model.support_vectors_
        all_results = {}
        # Results per vector
        perVector = (int)(num_results / len(sv)) + 2
        for vec in sv:
            # Return nearby images
            query_results = index.query(
                np.float32(vec),
                perVector,
                None,
                use_full_image,
                False,  # False bc just standard nearest neighbor
            )

            # Remove duplicates; for each image, include closest result
            for result in query_results:
                label = result.label
                if label not in all_results or all_results[label].dist > result.dist:
                    all_results[label] = result

        return resp.json(
            {
                "results": [r.to_dict() for r in all_results.values()],
                "svm_vector": utils.numpy_to_base64(w),
            }
        )  # unordered by distance for now
    else:
        return resp.json({"results": []})


# CLEANUP


@app.listener("after_server_stop")
async def cleanup(app, loop):
    print("Terminating:")
    await _cleanup_indexes()
    await _cleanup_clusters()


@utils.log_exception_from_coro_but_return_none
async def _cleanup_indexes():
    n = len(current_indexes)
    await current_indexes.clear_async()
    print(f"- cleaned up {n} indexes")


@utils.log_exception_from_coro_but_return_none
async def _cleanup_clusters():
    n = len(current_clusters)
    await current_clusters.clear_async()
    print(f"- killed {n} clusters")


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
