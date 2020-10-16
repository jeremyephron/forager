import asyncio
from collections import defaultdict
import functools
import heapq
import json
import operator

import numpy as np
from sanic import Sanic
import sanic.response as resp

from dataclasses import dataclass, field
from typing import Dict, Optional

from knn import utils
from knn.jobs import MapReduceJob, MapperSpec
from knn.reducers import Reducer, PoolingReducer
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
    service_url: Optional[str] = None


class LabeledIndex:
    def __init__(self, embedding_dict, **kwargs):
        self.labels = list(embedding_dict.keys())
        vectors = np.concatenate(
            [
                spatial_embeddings.reshape(config.EMBEDDING_DIM, -1).T
                for spatial_embeddings in embedding_dict.values()
            ]
        )
        ids = [
            i
            for i, (c, h, w) in enumerate(map(np.shape, embedding_dict.values()))
            for _ in range(h * w)
        ]

        self.index = InteractiveIndex(**kwargs)

        # Train
        # TODO(mihirg): Train only on a subset of the data
        self.index.train(vectors)

        # Add
        # TODO(mihirg): Add incrementally as results come back
        self.index.add(vectors, ids)

        self.index.merge_partial_indexes()

    def delete(self):
        self.index.cleanup()

    def query(self, query_vector, num_results, num_probes):
        dists, ids = self.index.query(
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


# Global data
current_clusters = {}  # type: Dict[str, ClusterData]
current_jobs = {}  # type: Dict[str, MapReduceJob]
current_indexes = {}  # type: Dict[str, LabeledIndex]


# Start web server
app = Sanic(__name__)
app.update_config({"RESPONSE_TIMEOUT": 10 * 60})  # 10 minutes


# CLUSTER MANAGEMENT
@app.route("/start_cluster", methods=["POST"])
async def start_cluster(request):
    n_nodes = int(request.form["n_nodes"][0])
    cluster = GKECluster(
        config.GCP_PROJECT, config.GCP_ZONE, config.GCP_MACHINE_TYPE, n_nodes
    )
    cluster_data = ClusterData(cluster, n_nodes)
    asyncio.create_task(_start_cluster(cluster_data))

    cluster_id = cluster.cluster_id
    current_clusters[cluster_id] = cluster_data
    return resp.json({"cluster_id": cluster_id})


async def _start_cluster(cluster_data):
    await cluster_data.cluster.start()
    cluster_data.started.set()

    deployment_id, service_url = await cluster_data.cluster.create_deployment(
        container=config.MAPPER_CONTAINER, num_replicas=cluster_data.n_nodes
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


async def _stop_cluster(cluster_data):
    await cluster_data.started.wait()
    await cluster_data.cluster.stop()


# EMBEDDING COMPUTATION
class EmbeddingDictReducer(Reducer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.embeddings = {}

    def handle_result(self, input, output):
        self.embeddings[input["image"]] = utils.base64_to_numpy(
            output[config.EMBEDDING_LAYER]
        )

    @property
    def result(self):
        return self.embeddings


@app.route("/start_job", methods=["POST"])
async def start_job(request):
    cluster_id = request.form["cluster_id"][0]
    bucket = request.form["bucket"][0]
    paths = request.form["paths"]

    cluster_data = current_clusters[cluster_id]
    await cluster_data.ready.wait()

    job = MapReduceJob(
        MapperSpec(url=cluster_data.service_url, n_mappers=cluster_data.n_nodes),
        EmbeddingDictReducer(),
        {"input_bucket": bucket},
        n_retries=config.N_RETRIES,
    )

    index_id = job.job_id
    current_jobs[index_id] = job

    # Construct input iterable
    iterable = [{"image": path} for path in paths]
    callback_func = functools.partial(_handle_job_result, index_id=index_id)
    await job.start(iterable, callback_func)

    return resp.json({"index_id": index_id})


def _handle_job_result(embedding_dict, index_id):
    # Create index
    current_indexes[index_id] = LabeledIndex(
        embedding_dict,
        tempdir=f"/tmp/{index_id}/",
        d=config.EMBEDDING_DIM,
        n_centroids=config.INDEX_NUM_CENTROIDS,
        vectors_per_index=config.INDEX_SUBINDEX_SIZE,
        use_gpu=config.INDEX_USE_GPU,
    )
    asyncio.create_task(_cleanup_job(index_id))


async def _cleanup_job(index_id):
    await asyncio.sleep(config.JOB_CLEANUP_TIME)
    current_jobs.pop(index_id, None)  # don't throw error if already deleted


@app.route("/job_status", methods=["GET"])
async def job_status(request):
    index_id = request.args["index_id"][0]

    if index_id in current_jobs:
        job = current_jobs[index_id]
        if job.finished:
            current_jobs.pop(index_id)
        result = job.job_result
    else:
        result = {}

    status = {
        "performance": result.get("performance"),
        "progress": result.get("progress"),
        "has_index": index_id in current_indexes,
    }
    return resp.json(status)


@app.route("/stop_job", methods=["POST"])
async def stop_job(request):
    index_id = request.json["index_id"]
    await current_jobs.pop(index_id).stop()
    return resp.text("", status=204)


# INDEX MANAGEMENT
def _extract_pooled_embedding_from_mapper_output(output):
    return utils.base64_to_numpy(output[config.EMBEDDING_LAYER]).mean(axis=(1, 2))


@app.route("/query_index", methods=["POST"])
async def query_index(request):
    cluster_id = request.form["cluster_id"][0]
    image_paths = request.form["paths"]
    bucket = request.form["bucket"][0]
    patches = [
        [float(patch[k]) for k in ("x1", "y1", "x2", "y2")]
        for patch in json.loads(request.form["patches"][0])
    ]  # [0, 1]^2
    index_id = request.form["index_id"][0]
    num_results = int(request.form["num_results"][0])

    cluster_data = current_clusters[cluster_id]
    await cluster_data.ready.wait()

    # Generate query vector as average of patch embeddings
    job = MapReduceJob(
        MapperSpec(url=cluster_data.service_url, n_mappers=1),
        PoolingReducer(extract_func=_extract_pooled_embedding_from_mapper_output),
        {"input_bucket": bucket},
        n_retries=config.N_RETRIES,
        chunk_size=1,
    )
    query_vector = await job.run_until_complete(
        [
            {"image": image_path, "patch": patch}
            for image_path, patch in zip(image_paths, patches)
        ]
    )

    # Run query and return results
    query_results = current_indexes[index_id].query(
        query_vector, num_results, config.INDEX_NUM_QUERY_PROBES
    )
    paths = [x[0] for x in query_results]
    return resp.json({"results": paths})


@app.route("/delete_index", methods=["POST"])
async def delete_index(request):
    index_id = request.form["index_id"][0]
    current_indexes.pop(index_id).delete()
    return resp.text("", status=204)


# CLEANUP
@app.listener("after_server_stop")
async def cleanup(app, loop):
    print("Terminating:")
    await _cleanup_jobs()
    await _cleanup_clusters()
    await _cleanup_indexes()


async def _cleanup_jobs():
    n = len(current_jobs)
    await asyncio.gather(*[job.stop() for job in current_jobs.values()])
    print(f"- stopped {n} jobs")


async def _cleanup_clusters():
    n = len(current_clusters)
    for cluster_data in current_clusters.values():
        await _stop_cluster(cluster_data)
    print(f"- killed {n} clusters")


async def _cleanup_indexes():
    n = len(current_indexes)
    for index in current_indexes.values():
        index.delete()
    print(f"- deleted {n} indexes")


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
