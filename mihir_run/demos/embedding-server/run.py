import asyncio
from collections import defaultdict
import functools
import gc
import heapq
import json
import operator
from pathlib import Path
from shutil import rmtree
import threading
import time
import uuid

import numpy as np
from sanic import Sanic
import sanic.response as resp
from sklearn import svm
from sklearn.metrics import accuracy_score

from dataclasses import dataclass, field

from typing import (
    Awaitable,
    Callable,
    Dict,
    DefaultDict,
    Set,
    TypeVar,
    Optional,
    MutableMapping,
)

from knn import utils
from knn.jobs import MapReduceJob, MapperSpec
from knn.reducers import Reducer, PoolingReducer, TrivialReducer
from knn.clusters import GKECluster

from interactive_index import InteractiveIndex

import config


@dataclass
class ClusterData:
    cluster: GKECluster
    n_nodes: int
    started: asyncio.Event = field(default_factory=asyncio.Event)
    ready: asyncio.Event = field(default_factory=asyncio.Event)
    deployment_id: Optional[str] = None
    n_replicas: Optional[int] = None
    service_url: Optional[str] = None


class LabeledIndexReducer(Reducer):
    LABEL_FILENAME = "labels.json"
    INDEX_PARENT_DIR = Path("~/forager/indexes").expanduser()

    FULL_INDEX_FOLDER = "f"
    SPATIAL_INDEX_FOLDER = "s"
    FULL_DOT_INDEX_FOLDER = "fd"
    SPATIAL_DOT_INDEX_FOLDER = "sd"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    # CONSTRUCTORS

    @classmethod
    def new(cls, *args, **kwargs) -> "LabeledIndexReducer":
        self = cls(*args, **kwargs)

        index_kwargs = dict(
            d=config.EMBEDDING_DIM,
            n_centroids=config.INDEX_NUM_CENTROIDS,
            vectors_per_index=config.INDEX_SUBINDEX_SIZE,
            use_gpu=config.INDEX_USE_GPU,
            transform=config.INDEX_TRANSFORM,
            transform_args=config.INDEX_TRANSFORM_ARGS,
            encoding=config.INDEX_ENCODING,
            encoding_args=config.INDEX_ENCODING_ARGS,
        )
        dot_index_kwargs = dict(**index_kwargs, metric="inner product")

        self.index_id = str(uuid.uuid4())
        self.index_dir = self.INDEX_PARENT_DIR / self.index_id
        self.labels = []
        self.full_index = InteractiveIndex(
            tempdir=str(self.index_dir / self.FULL_INDEX_FOLDER), **index_kwargs
        )
        self.spatial_index = InteractiveIndex(
            tempdir=str(self.index_dir / self.SPATIAL_INDEX_FOLDER), **index_kwargs
        )
        self.full_dot_index = InteractiveIndex(
            tempdir=str(self.index_dir / self.FULL_DOT_INDEX_FOLDER),
            **dot_index_kwargs,
        )
        self.spatial_dot_index = InteractiveIndex(
            tempdir=str(self.index_dir / self.SPATIAL_DOT_INDEX_FOLDER),
            **dot_index_kwargs,
        )

        self.accumulated_lock = threading.Lock()
        self.accumulated_full = {}
        self.accumulated_spatial = {}
        self.num_accumulated_spatial = 0

        self.should_finalize = threading.Event()
        self.flush_thread = threading.Thread(target=self.flush)
        self.flush_thread.start()

        return self

    @classmethod
    async def download(
        cls,
        index_id: str,
        *args,
        **kwargs,
    ) -> "LabeledIndexReducer":
        self = cls(*args, **kwargs)

        # Download from Cloud Storage
        self.INDEX_PARENT_DIR.mkdir(parents=True, exist_ok=True)
        # TODO(mihirg): Speed up - https://medium.com/@duhroach/gcs-read-performance-of-large-files-bd53cfca4410
        proc = await asyncio.create_subprocess_exec(
            "gsutil",
            "-m",
            "cp",
            "-r",
            "-n",
            f"{config.INDEX_UPLOAD_GCS_PATH}{index_id}",
            str(self.INDEX_PARENT_DIR),
        )
        await proc.wait()

        # Initialize indexes
        self.index_id = index_id
        self.index_dir = self.INDEX_PARENT_DIR / self.index_id
        self.labels = json.load((self.index_dir / self.LABEL_FILENAME).open())
        self.full_index = InteractiveIndex.load(
            str(self.index_dir / self.FULL_INDEX_FOLDER)
        )
        self.spatial_index = InteractiveIndex.load(
            str(self.index_dir / self.SPATIAL_INDEX_FOLDER)
        )
        self.full_dot_index = InteractiveIndex.load(
            str(self.index_dir / self.FULL_DOT_INDEX_FOLDER)
        )
        self.spatial_dot_index = InteractiveIndex.load(
            str(self.index_dir / self.SPATIAL_DOT_INDEX_FOLDER)
        )

        self.flush_thread = threading.Thread()  # dummy thread

        return self

    # REDUCER LIFECYCLE

    def handle_result(self, input, output):
        i = len(self.labels)

        self.labels.append(input["image"])
        embeddings = utils.base64_to_numpy(output[config.EMBEDDING_LAYER]).astype(
            np.float32
        )

        full = embeddings.mean(axis=(1, 2))
        spatial = embeddings.reshape(config.EMBEDDING_DIM, -1).T

        with self.accumulated_lock:
            self.accumulated_full[i] = full
            self.accumulated_spatial[i] = spatial
            self.num_accumulated_spatial += len(spatial)

    def flush(self):
        should_finalize = False

        while not should_finalize:
            should_finalize = self.should_finalize.is_set()

            accumulated_full_copy = {}
            accumulated_spatial_copy = {}

            with self.accumulated_lock:
                should_add_full = self.accumulated_full and (
                    should_finalize
                    or self.full_index.is_trained
                    or len(self.accumulated_full)
                    >= config.INDEX_TRAIN_MULTIPLE * self.full_index.n_centroids
                )
                should_add_spatial = self.accumulated_spatial and (
                    should_finalize
                    or self.spatial_index.is_trained
                    or self.num_accumulated_spatial
                    >= config.INDEX_TRAIN_MULTIPLE * self.spatial_index.n_centroids
                )

                # Swap local and global copies if necessary
                if should_add_full:
                    accumulated_full_copy, self.accumulated_full = (
                        self.accumulated_full,
                        accumulated_full_copy,
                    )
                if should_add_spatial:
                    (
                        accumulated_spatial_copy,
                        self.accumulated_spatial,
                        self.num_accumulated_spatial,
                    ) = (self.accumulated_spatial, accumulated_spatial_copy, 0)

            if should_add_full:
                full_vectors = np.stack(list(accumulated_full_copy.values()))
                full_ids = list(accumulated_full_copy.keys())

                if not self.full_index.is_trained:
                    self.full_index.train(full_vectors)
                if not self.full_dot_index.is_trained:
                    self.full_dot_index.train(full_vectors)
                self.full_index.add(full_vectors, full_ids)
                self.full_dot_index.add(full_vectors, full_ids)
                print(f"Added {len(full_ids)} full embeddings to index")

            if should_add_spatial:
                spatial_vectors = np.concatenate(
                    list(accumulated_spatial_copy.values())
                )
                spatial_ids = [
                    i
                    for i, vs in accumulated_spatial_copy.items()
                    for _ in range(len(vs))
                ]

                if not self.spatial_index.is_trained:
                    self.spatial_index.train(spatial_vectors)
                if not self.spatial_dot_index.is_trained:
                    self.spatial_dot_index.train(spatial_vectors)
                self.spatial_index.add(spatial_vectors, spatial_ids)
                self.spatial_dot_index.add(spatial_vectors, spatial_ids)
                print(f"Added {len(spatial_ids)} spatial embeddings to index")

            if not should_finalize and not should_add_full and not should_add_spatial:
                time.sleep(config.INDEX_FLUSH_SLEEP)
            else:
                gc.collect()

    @property
    def result(self):  # called only once to finalize
        self.should_finalize.set()
        self.flush_thread.join()

        for index in (
            self.full_index,
            self.spatial_index,
            self.full_dot_index,
            self.spatial_dot_index,
        ):
            index.merge_partial_indexes()
            index.delete_shards()

        return self

    # INDEX MANAGEMENT

    async def upload(self) -> None:
        # Dump labels
        json.dump(self.labels, (self.index_dir / self.LABEL_FILENAME).open("w"))

        # Upload to Cloud Storage
        # TODO(mihirg): Speed up - https://medium.com/google-cloud/google-cloud-storage-large-object-upload-speeds-7339751eaa24
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

    def delete(self) -> None:
        if self.flush_thread.is_alive():
            self.should_finalize.set()
            self.flush_thread.join()
        rmtree(self.index_dir)

    # QUERYING

    def query(
        self, query_vector, num_results, num_probes, use_full_image=False, svm=False
    ):
        assert not self.flush_thread.is_alive()

        if use_full_image:
            index = self.full_dot_index if svm else self.full_index
            dists, ids = index.query(query_vector, num_results, n_probes=num_probes)
            assert len(ids) == 1 and len(dists) == 1
            sorted_id_dist_tuples = [(i, d) for i, d in zip(ids[0], dists[0]) if i >= 0]
        else:
            index = self.spatial_dot_index if svm else self.spatial_index
            dists, ids = index.query(
                query_vector,
                config.QUERY_NUM_RESULTS_MULTIPLE * num_results,
                n_probes=num_probes,
            )
            assert len(ids) == 1 and len(dists) == 1

            # Gather lowest QUERY_PATCHES_PER_IMAGE distances for each image
            dists_by_id = defaultdict(list)
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

        return [(self.labels[i], d) for i, d in sorted_id_dist_tuples]


KT = TypeVar("KT")
VT = TypeVar("VT")


class ExpiringDict(MutableMapping[KT, VT]):
    def __init__(
        self,
        sanic_app: Sanic,
        cleanup_func: Callable[[VT], Awaitable[None]],
        timeout: Optional[float] = None,
        interval: float = 1.0,
    ) -> None:
        self.schedule = sanic_app.add_task
        self.cleanup_func = cleanup_func
        self.timeout = timeout
        self.interval = interval

        self.store: Dict[KT, VT] = {}
        self.last_accessed: Dict[KT, float] = {}
        self.locks: DefaultDict[KT, Set[str]] = defaultdict(set)

        self.schedule(self.sleep_and_cleanup())

    def __getitem__(self, key: KT) -> VT:
        value = self.store[key]
        self.last_accessed[key] = time.time()
        return value

    def __setitem__(self, key: KT, value: VT) -> None:
        self.store[key] = value
        self.last_accessed[key] = time.time()

    def __delitem__(self, key: KT) -> None:
        del self.store[key]
        del self.last_accessed[key]
        del self.locks[key]

    def __iter__(self):
        raise NotImplementedError()

    def __len__(self) -> int:
        return len(self.store)

    def lock(self, key: KT, lock_name: str) -> None:
        self.locks[key].add(lock_name)

    def unlock(self, key: KT, lock_name: str) -> None:
        self.last_accessed[key] = time.time()
        if lock_name in self.locks[key]:
            self.locks[key].remove(lock_name)

    def clear(self) -> None:
        asyncio.run(self.clear_async())

    async def cleanup_key(self, key: KT) -> None:
        await self.cleanup_func(self.store.pop(key))
        del self.last_accessed[key]

    async def clear_async(self) -> None:
        self.locks.clear()
        await asyncio.gather(*map(self.cleanup_key, self.store.keys()))

    async def sleep_and_cleanup(self) -> None:
        if not self.timeout:
            return

        await asyncio.sleep(self.interval)

        current = time.time()
        keys_to_delete = [
            k
            for k, v in self.last_accessed.items()
            if current - v > self.timeout and not self.locks[k]
        ]
        await asyncio.gather(*map(self.cleanup_key, keys_to_delete))

        self.schedule(self.sleep_and_cleanup())


# Start web server
app = Sanic(__name__)
app.update_config({"RESPONSE_TIMEOUT": 10 * 60})  # 10 minutes


async def _stop_cluster(cluster_data):
    await cluster_data.started.wait()
    await cluster_data.cluster.stop()


async def _stop_job(job):
    await job.stop()
    current_clusters.unlock(job.cluster_id, job.job_id)
    job.reducer.delete()


async def _delete_index(index):
    index.delete()


# Global data
current_clusters = ExpiringDict(
    app, _stop_cluster, config.CLEANUP_TIMEOUT, config.CLEANUP_INTERVAL
)  # type: ExpiringDict[str, ClusterData]
current_jobs = ExpiringDict(
    app, _stop_job, config.CLEANUP_TIMEOUT, config.CLEANUP_INTERVAL
)  # type: ExpiringDict[str, MapReduceJob]
current_indexes = ExpiringDict(  # never evict (for now)
    app, _delete_index  # , config.CLEANUP_TIMEOUT, config.CLEANUP_INTERVAL
)  # type: ExpiringDict[str, LabeledIndexReducer]


# CLUSTER MANAGEMENT
@app.route("/start_cluster", methods=["POST"])
async def start_cluster(request):
    n_nodes = int(request.form["n_nodes"][0])
    cluster = GKECluster(
        config.GCP_PROJECT, config.GCP_ZONE, config.GCP_MACHINE_TYPE, n_nodes
    )
    cluster_data = ClusterData(cluster, n_nodes)
    app.add_task(_start_cluster(cluster_data))

    cluster_id = cluster.cluster_id
    current_clusters[cluster_id] = cluster_data
    return resp.json({"cluster_id": cluster_id})


async def _start_cluster(cluster_data):
    await cluster_data.cluster.start()
    cluster_data.started.set()

    cluster_data.n_replicas = cluster_data.n_nodes * config.CLUSTER_NODE_N_MAPPERS
    deployment_id, service_url = await cluster_data.cluster.create_deployment(
        container=config.MAPPER_CONTAINER,
        num_replicas=cluster_data.n_replicas,
    )
    cluster_data.deployment_id = deployment_id
    cluster_data.service_url = service_url
    cluster_data.ready.set()


@app.route("/cluster_status", methods=["GET"])
async def cluster_status(request):
    cluster_id = request.args["cluster_id"][0]
    cluster_data = current_clusters.get(cluster_id)
    has_cluster = cluster_data is not None

    status = {
        "has_cluster": has_cluster,
        "started": has_cluster and cluster_data.started.is_set(),
        "ready": has_cluster and cluster_data.ready.is_set(),
    }
    return resp.json(status)


@app.route("/stop_cluster", methods=["POST"])
async def stop_cluster(request):
    cluster_id = request.form["cluster_id"][0]
    cluster_data = current_clusters.pop(cluster_id)
    await _stop_cluster(cluster_data)
    return resp.text("", status=204)


# EMBEDDING COMPUTATION
@app.route("/start_job", methods=["POST"])
async def start_job(request):
    cluster_id = request.form["cluster_id"][0]
    bucket = request.form["bucket"][0]
    paths = request.form["paths"]

    cluster_data = current_clusters[cluster_id]
    await cluster_data.ready.wait()

    index = LabeledIndexReducer.new()
    job = MapReduceJob(
        MapperSpec(
            url=cluster_data.service_url,
            n_mappers=cluster_data.n_replicas,
        ),
        index,
        {"input_bucket": bucket},
        n_retries=config.N_RETRIES,
        chunk_size=config.CHUNK_SIZE,
    )
    job.cluster_id = cluster_id

    index_id = index.index_id
    current_jobs[index_id] = job

    # Lock cluster to prevent deletion during long-running jobs
    job_id = job.job_id
    current_clusters.lock(cluster_id, job_id)

    # Construct input iterable
    augmentation_dict = {}
    iterable = [{"image": path, "augmentations": augmentation_dict} for path in paths]
    callback_func = functools.partial(
        _handle_job_result, index_id=index_id, cluster_id=cluster_id, job_id=job_id
    )

    await job.start(iterable, callback_func)

    return resp.json({"index_id": index_id})


async def _handle_job_result(index, index_id, cluster_id, job_id):
    await index.upload()
    current_indexes[index_id] = index
    current_clusters.unlock(cluster_id, job_id)


@app.route("/job_status", methods=["GET"])
async def job_status(request):
    index_id = request.args["index_id"][0]

    if index_id in current_jobs:
        job = current_jobs[index_id]
        if job.finished:
            current_jobs.pop(index_id)
        performance = job.performance
        progress = job.progress
    else:
        performance = None
        progress = None

    status = {
        "performance": performance,
        "progress": progress,
        "has_index": index_id in current_indexes,
    }
    return resp.json(status)


@app.route("/stop_job", methods=["POST"])
async def stop_job(request):
    index_id = request.json["index_id"]
    await _stop_job(current_jobs.pop(index_id))
    return resp.text("", status=204)


# INDEX MANAGEMENT


def _extract_pooled_embedding_from_mapper_output(output):
    return (
        utils.base64_to_numpy(output[config.EMBEDDING_LAYER])
        .mean(axis=(1, 2))
        .astype(np.float32)
    )


async def _download_index(index_id):
    index = await LabeledIndexReducer.download(index_id)
    current_indexes[index_id] = index


@app.route("/download_index", methods=["POST"])
async def download_index(request):
    index_id = request.form["index_id"][0]
    app.add_task(_download_index(index_id))
    return resp.text("", status=204)


@app.route("/delete_index", methods=["POST"])
async def delete_index(request):
    index_id = request.form["index_id"][0]
    await _delete_index(current_indexes.pop(index_id))
    return resp.text("", status=204)


@app.route("/query_index", methods=["POST"])
async def query_index(request):
    cluster_id = request.form["cluster_id"][0]
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

    use_full_image = bool(request.form.get("use_full_image", [False])[0])

    if cluster_id in current_clusters:
        cluster_data = current_clusters[cluster_id]
        await cluster_data.ready.wait()
        mapper_url = cluster_data.service_url
        n_mappers = cluster_data.n_replicas
    else:
        mapper_url = config.MAPPER_CLOUD_RUN_URL
        n_mappers = config.CLOUD_RUN_N_MAPPERS

    # Generate query vector as average of patch embeddings
    job = MapReduceJob(
        MapperSpec(url=mapper_url, n_mappers=n_mappers),
        PoolingReducer(extract_func=_extract_pooled_embedding_from_mapper_output),
        {"input_bucket": bucket},
        n_retries=config.N_RETRIES,
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
        query_vector, num_results, config.INDEX_NUM_QUERY_PROBES, use_full_image, False
    )
    paths = [x[0] for x in query_results]
    return resp.json({"results": paths})


@app.route("/active_batch", methods=["POST"])
async def active_batch(request):
    cluster_id = request.form["cluster_id"][0]
    image_paths = request.form["paths"]  # Paths of seed images (at first from google)
    # patches = [
    #    [float(patch[k]) for k in ("x1", "y1", "x2", "y2")]
    #    for patch in json.loads(request.form["patches"][0]) # Modified to pass all patches, not just the first
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

    cluster_data = current_clusters[cluster_id]
    await cluster_data.ready.wait()

    # Generate query vector as average of patch embeddings
    job = MapReduceJob(
        MapperSpec(
            url=cluster_data.service_url,
            n_mappers=cluster_data.n_replicas,
        ),
        TrivialReducer(extract_func=_extract_pooled_embedding_from_mapper_output),
        {"input_bucket": bucket},
        n_retries=config.N_RETRIES,
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
            config.INDEX_NUM_QUERY_PROBES,
            use_full_image,
            False,  # False bc just standard nearest neighbor
        )
        paths.extend([x[0] for x in query_results])
    paths = list(set(paths))  # Unordered, could choose to order this by distance

    return resp.json({"results": paths})


# SVM


@app.route("/query_svm", methods=["POST"])
async def query_svm(request):
    cluster_id = request.form["cluster_id"][0]
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

    cluster_data = current_clusters[cluster_id]
    await cluster_data.ready.wait()

    # Get embeddings from index
    # We may want to get patch embeddings for labeled images in future--might as well use the bounding box
    # Only do this if we are using spatial KNN?
    # Can't really do the following lines on a large index--too expensive
    # embedding_keys = current_indexes[index_id].labels
    # embedding = current_indexes[index_id].values

    # Generate training vectors
    job = MapReduceJob(
        MapperSpec(
            url=cluster_data.service_url,
            n_mappers=cluster_data.n_replicas,
        ),  # Figure out n_mappers later
        TrivialReducer(
            extract_func=_extract_pooled_embedding_from_mapper_output
        ),  # Returns all individual inputs back
        {"input_bucket": bucket},
        n_retries=config.N_RETRIES,
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
            url=cluster_data.service_url,
            n_mappers=cluster_data.n_replicas,
        ),  # Figure out n_mappers later
        TrivialReducer(
            extract_func=_extract_pooled_embedding_from_mapper_output
        ),  # Returns all individual inputs back
        {"input_bucket": bucket},
        n_retries=config.N_RETRIES,
        chunk_size=1,
    )
    neg_features = await job.run_until_complete(neg_inputs)
    training_labels = np.concatenate(
        [
            np.ones(
                (
                    len(
                        pos_inputs,
                    )
                ),
                dtype=int,
            ),
            np.zeros(
                (
                    len(
                        neg_inputs,
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

    use_full_image = bool(request.form.get("use_full_image", [False])[0])

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
            w, num_results, config.INDEX_NUM_QUERY_PROBES, use_full_image, True
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
                config.INDEX_NUM_QUERY_PROBES,
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
    await _cleanup_jobs()
    await _cleanup_clusters()
    await _cleanup_indexes()


async def _cleanup_jobs():
    n = len(current_jobs)
    await current_jobs.clear_async()
    print(f"- stopped {n} jobs")


async def _cleanup_clusters():
    n = len(current_clusters)
    await current_clusters.clear_async()
    print(f"- killed {n} clusters")


async def _cleanup_indexes():
    pass  # never evict (for now)
    # n = len(current_indexes)
    # await current_indexes.clear_async()
    # print(f"- deleted {n} indexes")


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
