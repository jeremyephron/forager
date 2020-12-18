import asyncio
import concurrent
from collections import defaultdict
from distutils.util import strtobool
import functools
import heapq
import json
import logging
import operator
import os
from pathlib import Path
import shutil
import random
import time
import uuid

import aiohttp
import numpy as np
from sanic import Sanic
import sanic.response as resp
from sklearn import svm
from sklearn.metrics import accuracy_score

from typing import DefaultDict, Dict, List, Optional, Iterable

from interactive_index import InteractiveIndex

from knn.clusters import TerraformModule
from knn.jobs import MapReduceJob, MapperSpec
from knn.reducers import PoolingReducer, TrivialReducer
from knn.utils import JSONType, unasync_as_task

import config
from index_jobs import (
    AdderReducer,
    IndexType,
    MapperReducer,
    TrainingJob,
    extract_pooled_embedding_from_mapper_output,
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
log_ch.setLevel(logging.ERROR)

# create formatter and add it to the handlers
formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
log_fh.setFormatter(formatter)
log_ch.setFormatter(formatter)

# Attach handlers
logger.addHandler(log_fh)
logger.addHandler(log_ch)


class LabeledIndex:
    LABEL_FILENAME = "labels.json"

    # Don't use this directly - use a @classmethod constructor
    def __init__(self, *args, **kwargs):
        self.logger = logging.getLogger("index_server.LabeledIndex")
        self.ready = asyncio.Event()

        # Will be filled by each individual constructor
        self.index_id: Optional[str] = None
        self.index_dir: Optional[Path] = None
        self.labels: List[str] = []
        self.indexes: Dict[IndexType, InteractiveIndex] = {}

        # Will only be used by the start_building() pathway
        self.cluster: Optional[TerraformModule] = None
        self.session: Optional[aiohttp.ClientSession] = None
        self.mapper_job: Optional[MapReduceJob] = None
        self.training_jobs: CleanupDict[IndexType, TrainingJob] = CleanupDict(
            lambda job: job.stop()
        )
        self.start_adding_eventually_task: Optional[asyncio.Task] = None
        self.adder_job: Optional[MapReduceJob] = None

    def delete(self):
        # TODO(mihirg): More checks here?
        shutil.rmtree(self.index_dir)

    def query(
        self,
        query_vector: np.ndarray,
        num_results: int,
        num_probes: Optional[int] = None,
        use_full_image: bool = False,
        svm: bool = False,
    ):
        assert self.ready.is_set()

        self.logger.info(f"Query: use_full_image={use_full_image}, svm={svm}")
        start = time.perf_counter()

        if use_full_image:
            index = self.indexes[IndexType.FULL_DOT if svm else IndexType.FULL]
            dists, ids = index.query(query_vector, num_results, n_probes=num_probes)
            assert len(ids) == 1 and len(dists) == 1
            sorted_id_dist_tuples = [(i, d) for i, d in zip(ids[0], dists[0]) if i >= 0]
        else:
            index = self.indexes[IndexType.SPATIAL_DOT if svm else IndexType.SPATIAL]
            dists, ids = index.query(
                query_vector,
                config.QUERY_NUM_RESULTS_MULTIPLE * num_results,
                n_probes=num_probes,
            )
            assert len(ids) == 1 and len(dists) == 1

            # Gather lowest QUERY_PATCHES_PER_IMAGE distances for each image
            dists_by_id: DefaultDict[int, List[float]] = defaultdict(list)
            for i, d in zip(ids[0], dists[0]):
                if i >= 0 and len(dists_by_id[i]) < config.QUERY_PATCHES_PER_IMAGE:
                    dists_by_id[i].append(d)

            # Average them and resort
            id_dist_tuple_gen = (
                (i, sum(ds) / len(ds))
                for i, ds in dists_by_id.items()
                if len(ds) == config.QUERY_PATCHES_PER_IMAGE
            )
            sorted_id_dist_tuples = heapq.nsmallest(
                num_results, id_dist_tuple_gen, operator.itemgetter(1)
            )

        end = time.perf_counter()
        self.logger.debug(
            f"Query of size {query_vector.shape} with k={num_results}, "
            f"n_probes={num_probes}, n_centroids={index.n_centroids}, and "
            f"n_vectors={index.n_vectors} took {end - start:.3f}s, and "
            f"got {len(sorted_id_dist_tuples)} results."
        )

        return [(self.labels[i], d) for i, d in sorted_id_dist_tuples]

    # INDEX CREATION

    @classmethod
    async def start_building(
        cls,
        cluster: TerraformModule,
        bucket: str,
        paths: List[str],
        *args,
        **kwargs,
    ):
        self = cls(*args, **kwargs)

        await asyncio.gather(cluster.ready.wait(), cluster.mounted.wait())
        self.cluster = cluster

        self.index_id = str(uuid.uuid4())
        self.index_dir = config.INDEX_PARENT_DIR / self.index_id
        self.index_dir.mkdir(parents=True, exist_ok=False)

        self.training_http_session = aiohttp.ClientSession()
        for index_type in IndexType:
            self.training_jobs[index_type] = TrainingJob(
                index_type,
                len(paths),
                self.index_id,
                self.cluster.output["trainer_urls"][index_type.value],
                self.cluster.mount_dir,
                self.training_http_session,
            )
        notification_requests_to_start_training = [
            job.make_notification_request_to_start_training(
                functools.partial(self.start_training, index_type=index_type)
            )
            for index_type, job in self.training_jobs.items()
        ]

        # Randomly shuffle input images
        self.labels = random.sample(paths, k=len(paths))
        iterable = (
            {"id": i, "image": path, "augmentations": {}}
            for i, path in enumerate(self.labels)
        )

        # Step 1: "Map" input images to embedding files saved to shared disk
        self.mapper_job = MapReduceJob(
            MapperSpec(
                url=self.cluster.output["mapper_url"],
                n_mappers=self.cluster.output["num_mappers"],
            ),
            MapperReducer(notification_requests_to_start_training),
            {"input_bucket": bucket, "return_type": "save"},
            n_retries=config.MAPPER_NUM_RETRIES,
            chunk_size=config.MAPPER_CHUNK_SIZE,
            request_timeout=config.MAPPER_REQUEST_TIMEOUT,
        )
        await self.mapper_job.start(iterable, self.start_training, len(paths))

        # Start a background task that waits until the Train step is done and then
        # kicks off the Add step
        self.start_adding_eventually_task = asyncio.create_task(
            self.start_adding_eventually()
        )  # background task

        return self

    def start_training(
        self,
        output_paths: List[str],
        index_type: Optional[IndexType] = None,
    ):
        # Step 2: "Train" each index once we have enough images/spatial embeddings (or
        # when the Map step finishes, in which case index_type=None indicating that we
        # should train all remaining indexes)
        index_types = [index_type] if index_type else iter(IndexType)
        for index_type in index_types:
            if self.training_jobs[index_type].started:
                continue
            asyncio.create_task(self.training_jobs[index_type].start(output_paths))

    async def handle_training_status_update(self, result: JSONType):
        # Because training takes a long time, the trainer sends us back an HTTP request
        # on status changes rather than communicating over a long-standing request.
        # This function is called by the Sanic endpoint and passes the status update
        # along to the relevant training job.
        print("A", result)
        index_type = IndexType[result["index_name"]]
        if index_type in self.training_jobs:
            await self.training_jobs[index_type].handle_result(result)

    async def start_adding_eventually(self):
        # Wait until all indexes are trained
        indexes = {}
        for index_type, job in self.training_jobs.items():
            await job.finished.wait()
            indexes[index_type.name] = {
                "average": job.average,
                "index_dir": job.index_dir,
            }
            print(f"Job for index {index_type.name} finished")

            # Copy index training results to local disk before anything else gets
            # written into the index directory on the shared disk
            shutil.copytree(job.mounted_index_dir, self.index_dir)
            print(f"Copied {job.mounted_index_dir} to {self.index_dir}")

        # TODO(mihirg): Fix this!
        print("Before wait")
        await self.mapper_job.reducer.finished.wait()
        print("After wait")

        # Step 3: As the Map step computes and saves embeddings, "Add" them into shards
        # of the newly trained indexes
        self.adder_job = MapReduceJob(
            MapperSpec(
                url=self.cluster.output["adder_url"],
                n_mappers=self.cluster.output["num_adders"],
            ),
            AdderReducer(),
            {"indexes": indexes},
            n_retries=config.ADDER_NUM_RETRIES,
            chunk_size=config.ADDER_CHUNK_SIZE,
            request_timeout=config.ADDER_REQUEST_TIMEOUT,
        )
        await self.adder_job.start(self.mapper_job.reducer.result, self.finished_adding)
        # await self.adder_job.start(
        #     self.mapper_job.reducer.output_paths_gen(), self.finished_adding
        # )  # iterable is an async generator that yields as the Map step produces outputs

    async def stop_building(self):
        # Map
        if self.mapper_job:
            await self.mapper_job.stop()

        # Train
        await self.training_jobs.clear_async()
        await self.training_http_session.close()

        # Add
        # TODO(mihirg): Also stop the task from finished_adding()?
        if (
            self.start_adding_eventually_task
            and not self.start_adding_eventually_task.done()
        ):
            self.start_adding_eventually_task.cancel()
            await self.start_adding_eventually_task
        if self.adder_job:
            await self.adder_job.stop()

    @unasync_as_task
    async def finished_adding(self, shard_tmpls: Iterable[str]):
        # loop = asyncio.get_running_loop()

        # Merge shards from shared disk into local index
        # TODO(mihirg): Consider deleting all unnecessary intermediates from NAS after
        self._load_local_indexes()
        # with concurrent.futures.ProcessPoolExecutor(max_workers=1) as pool:
        for index_type, index in self.indexes.items():
            index_dir = self.training_jobs[index_type].mounted_index_dir
            shard_paths = [
                str(p.resolve())
                for shard_tmpl in shard_tmpls
                for p in index_dir.glob(shard_tmpl.format("*"))
            ]
            index.merge_partial_indexes(shard_paths)

            # await loop.run_in_executor(
            #     pool, functools.partial(index.merge_partial_indexes, shard_paths)
            # )

        await self.upload()
        self.ready.set()

    async def upload(self):
        # Dump labels
        json.dump(self.labels, (self.index_dir / self.LABEL_FILENAME).open("w"))

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
        # TODO(mihirg): Finish filling this in
        return {
            "map_progress": self.mapper_job.progress if self.mapper_job else {},
            "train_progress": {
                index_type.name: {
                    "started": job.started,
                    "finished": job.finished.is_set(),
                }
                for index_type, job in self.training_jobs.items()
            },
            "add_progress": self.adder_job.progress if self.adder_job else {},
        }

    # INDEX LOADING

    @classmethod
    async def download(
        cls,
        index_id: str,
        *args,
        **kwargs,
    ) -> "LabeledIndex":
        self = cls(*args, **kwargs)

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
            f"{config.INDEX_UPLOAD_GCS_PATH}{index_id}",
            str(config.INDEX_PARENT_DIR),
        )
        await proc.wait()

        # Initialize indexes
        self.index_id = index_id
        self.index_dir = config.INDEX_PARENT_DIR / self.index_id
        self.labels = json.load((self.index_dir / self.LABEL_FILENAME).open())
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
    # Hack(mihirg): just attach mounted and mount_dir attributes to the cluster object
    cluster.mounted = asyncio.Event()
    await cluster.apply()

    # Mount NFS
    cluster.mount_dir = (
        config.CLUSTER_MOUNT_DIR
        / cluster.id
        / cluster.output["nfs_mount_dir"].lstrip(os.sep)
    )
    cluster.mount_dir.mkdir(parents=True, exist_ok=True)
    proc = await asyncio.create_subprocess_exec(
        "sudo",
        "mount",
        cluster.output["nfs_url"],
        str(cluster.mount_dir),
    )
    await proc.wait()
    cluster.mounted.set()


async def _stop_cluster(cluster):
    # Unmount NFS
    await cluster.mounted.wait()
    proc = await asyncio.create_subprocess_exec(
        "sudo", "umount", str(cluster.mount_dir)
    )
    await proc.wait()
    shutil.rmtree(config.CLUSTER_MOUNT_DIR / cluster.id)

    # Destroy cluster
    # await cluster.ready.wait()
    # await cluster.destroy()


# TODO(mihirg): Automatically clean up inactive clusters
current_clusters: CleanupDict[str, TerraformModule] = CleanupDict(_stop_cluster)


@app.route("/start_cluster", methods=["POST"])
async def start_cluster(request):
    cluster = TerraformModule(config.CLUSTER_TERRAFORM_MODULE_PATH, copy=False)
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
# TODO(mihirg): Change input from form encoding to JSON


@app.route("/start_job", methods=["POST"])
async def start_job(request):
    cluster_id = request.form["cluster_id"][0]
    bucket = request.form["bucket"][0]
    paths = request.form["paths"]

    cluster = current_clusters[cluster_id]
    index = await LabeledIndex.start_building(cluster, bucket, paths)

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


@app.route("/query_index", methods=["POST"])
async def query_index(request):
    image_paths = request.form["paths"]
    bucket = request.form["bucket"][0]
    patches = [
        [float(patch[k]) for k in ("x1", "y1", "x2", "y2")]
        for patch in json.loads(
            request.form["patches"][0]
        )  # Modified to pass all patches, not just the first
    ]  # [0, 1]^2
    print(patches)
    index_id = request.form["index_id"][0]
    num_results = int(request.form["num_results"][0])
    augmentations = []
    if "augmentations" in request.form:
        augmentations = request.form["augmentations"]

    augmentation_dict = {}
    for i in range(len(augmentations) // 2):
        augmentation_dict[augmentations[2 * i]] = float(augmentations[2 * i + 1])

    use_full_image = bool(strtobool(request.form.get("use_full_image", "False")))

    # TODO(mihirg): Fall back to cluster?
    mapper_url = config.MAPPER_CLOUD_RUN_URL
    n_mappers = config.CLOUD_RUN_N_MAPPERS

    # Generate query vector as average of patch embeddings
    job = MapReduceJob(
        MapperSpec(url=mapper_url, n_mappers=n_mappers),
        PoolingReducer(extract_func=extract_pooled_embedding_from_mapper_output),
        {"input_bucket": bucket},
        n_retries=config.CLOUD_RUN_N_RETRIES,
        chunk_size=1,
    )
    query_vector = await job.run_until_complete(
        [
            {"image": image_path, "patch": patch, "augmentations": augmentation_dict}
            for image_path, patch in zip(image_paths, patches)
        ]
    )

    # Run query and return results
    query_results = current_indexes[index_id].query(
        query_vector, num_results, None, use_full_image, False
    )
    paths = [x[0] for x in query_results]
    return resp.json({"results": paths})


@app.route("/active_batch", methods=["POST"])
async def active_batch(request):
    image_paths = request.form["paths"]  # Paths of seed images (at first from google)
    # patches = [
    #    [float(patch[k]) for k in ('x1', 'y1', 'x2', 'y2')]
    #    for patch in json.loads(request.form['patches'][0]) # Modified to pass all patches, not just the first
    # ]  # [0, 1]^2
    bucket = request.form["bucket"][0]
    index_id = request.form["index_id"][0]
    num_results = int(request.form["num_results"][0])
    augmentations = []
    if "augmentations" in request.form:
        augmentations = request.form["augmentations"]

    augmentation_dict = {}
    for i in range(len(augmentations) // 2):
        augmentation_dict[augmentations[2 * i]] = float(augmentations[2 * i + 1])

    use_full_image = True

    # TODO(mihirg): Fall back to cluster?
    mapper_url = config.MAPPER_CLOUD_RUN_URL
    n_mappers = config.CLOUD_RUN_N_MAPPERS

    # Generate query vector as average of patch embeddings
    job = MapReduceJob(
        MapperSpec(
            url=mapper_url,
            n_mappers=n_mappers,
        ),
        TrivialReducer(extract_func=extract_pooled_embedding_from_mapper_output),
        {"input_bucket": bucket},
        n_retries=config.CLOUD_RUN_N_RETRIES,
        chunk_size=1,
    )
    query_vectors = await job.run_until_complete(
        [
            {"image": image_path, "augmentations": augmentation_dict}
            for image_path in image_paths
        ]
    )

    paths = []
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
        paths.extend([x[0] for x in query_results])
    paths = list(set(paths))  # Unordered, could choose to order this by distance

    return resp.json({"results": paths})


@app.route("/query_svm", methods=["POST"])
async def query_svm(request):
    index_id = request.form["index_id"][0]
    bucket = request.form["bucket"][0]
    pos_image_paths = request.form["positive_paths"]
    pos_patches = [
        [float(patch[k]) for k in ("x1", "y1", "x2", "y2")]
        for patch in json.loads(
            request.form["positive_patches"][0]
        )  # Pass all patches, not just the first
    ]  # [0, 1]^2
    neg_image_paths = request.form["negative_paths"]
    num_results = int(request.form["num_results"][0])
    mode = request.form["mode"][0]

    # TODO(mihirg): Fall back to cluster?
    mapper_url = config.MAPPER_CLOUD_RUN_URL
    n_mappers = config.CLOUD_RUN_N_MAPPERS

    # Get embeddings from index
    # We may want to get patch embeddings for labeled images in future--might as well use the bounding box
    # Only do this if we are using spatial KNN?
    # Can't really do the following lines on a large index--too expensive
    # embedding_keys = current_indexes[index_id].labels
    # embedding = current_indexes[index_id].values

    # Generate training vectors
    job = MapReduceJob(
        MapperSpec(
            url=mapper_url,
            n_mappers=n_mappers,
        ),  # Figure out n_mappers later
        TrivialReducer(
            extract_func=extract_pooled_embedding_from_mapper_output
        ),  # Returns all individual inputs back
        {"input_bucket": bucket},
        n_retries=config.CLOUD_RUN_N_RETRIES,
        chunk_size=1,
    )
    augmentation_dict = {}
    pos_inputs = [
        {"image": image_path, "patch": patch, "augmentations": augmentation_dict}
        for image_path, patch in zip(pos_image_paths, pos_patches)
    ]
    neg_inputs = [
        {"image": image_path, "augmentations": augmentation_dict}
        for image_path in neg_image_paths
    ]
    # Run positive and negatives separately in case the reducer doesn't maintain order
    print(pos_inputs + neg_inputs)
    pos_features = await job.run_until_complete(pos_inputs)
    job = MapReduceJob(
        MapperSpec(
            url=mapper_url,
            n_mappers=n_mappers,
        ),  # Figure out n_mappers later
        TrivialReducer(
            extract_func=extract_pooled_embedding_from_mapper_output
        ),  # Returns all individual inputs back
        {"input_bucket": bucket},
        n_retries=config.CLOUD_RUN_N_RETRIES,
        chunk_size=1,
    )
    neg_features = await job.run_until_complete(neg_inputs)
    training_labels = np.concatenate(
        [
            np.ones(
                (
                    len(
                        pos_features,
                    )
                ),
                dtype=int,
            ),
            np.zeros(
                (
                    len(
                        neg_features,
                    )
                ),
                dtype=int,
            ),
        ]
    )
    training_features = np.concatenate((pos_features, neg_features), axis=0)

    # Train the SVM using pos/neg + their corresponding embeddings
    model = svm.SVC(kernel="linear")
    model.fit(training_features, training_labels)
    predicted = model.predict(training_features)

    # get the accuracy
    print(accuracy_score(training_labels, predicted))

    use_full_image = bool(strtobool(request.form.get("use_full_image", "False")))

    if mode == "svmPos" or mode == "spatialSvmPos":
        # Evaluate the SVM by querying index
        w = model.coef_  # This will be the query vector
        # Also consider returning the support vectors--good to look at examples along hyperplane
        w = np.float32(w[0] * 1000)

        augmentations = []
        if "augmentations" in request.form:
            augmentations = request.form["augmentations"]

        augmentation_dict = {}
        for i in range(len(augmentations) // 2):
            augmentation_dict[augmentations[2 * i]] = float(augmentations[2 * i + 1])

        # Run query and return results
        query_results = current_indexes[index_id].query(
            w, num_results, None, use_full_image, True
        )

        paths = [x[0] for x in query_results]
        return resp.json({"results": paths})
    elif mode == "svmBoundary":
        # Get samples close to boundary vectors
        # For now, looks like most vectors end up being support vectors since underparamtrized system
        sv = model.support_vectors_
        paths = []
        # Results per vector
        perVector = (int)(num_results / len(sv)) + 2
        for vec in sv:
            # Return nearby images
            query_results = current_indexes[index_id].query(
                np.float32(vec),
                perVector,
                None,
                use_full_image,
                False,  # False bc just standard nearest neighbor
            )
            paths.extend([x[0] for x in query_results])
        # Remove duplicates (without maintaining order for now)
        paths = list(set(paths))
        return resp.json({"results": paths})
    else:
        return resp.json({"results": []})


# CLEANUP


@app.listener("after_server_stop")
async def cleanup(app, loop):
    print("Terminating:")
    await _cleanup_clusters()
    await _cleanup_index_build_jobs()


async def _cleanup_index_build_jobs():
    n = len(current_indexes)
    await current_indexes.clear_async()
    print(f"- cleaned up {n} index build jobs")


async def _cleanup_clusters():
    n = len(current_clusters)
    await current_clusters.clear_async()
    print(f"- killed {n} clusters")


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
