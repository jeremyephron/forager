import asyncio
import functools
import uuid

import numpy as np

from typing import Dict

from jinja2 import Environment, FileSystemLoader, select_autoescape
from sanic import Sanic
from sanic.response import json, html, text

from knn.jobs import MapReduceJob, MapperSpec
from knn.reducers import Reducer
from knn.utils import FileListIterator, base64_to_numpy
from knn.clusters import GKECluster

from interactive_index import InteractiveIndex

import config


class EmbeddingDictReducer(Reducer):
    def __init__(self, embedding_layer, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.embedding_layer = embedding_layer
        self.embeddings = {}

    def handle_result(self, input, output):
        self.embeddings[input] = base64_to_numpy(output[self.embedding_layer])

    @property
    def result(self):
        return self.embeddings


# Start web server
app = Sanic(__name__)
app.static("/static", "./static")
app.update_config({"RESPONSE_TIMEOUT": 10 * 60})  # 10 minutes
jinja = Environment(
    loader=FileSystemLoader("./templates"),
    autoescape=select_autoescape(["html"]),
)

current_clusters = {}  # type: Dict[str, GKECluster]
current_queries = {}  # type: Dict[str, MapReduceJob]


@app.route("/")
async def homepage(request):
    template = jinja.get_template("index.html")
    response = template.render(
        n_concurrent_workers=config.N_CONCURRENT_WORKERS_DEFAULT,
        demo_images=config.DEMO_IMAGES,
        image_bucket=config.IMAGE_BUCKET,
    )
    return html(response)


@app.route("/start_cluster", methods=["POST"])
async def start_cluster(request):
    n_nodes = int(request.form["n_nodes"])

    gke_cluster = GKECluster(
        config.GCP_PROJECT, config.GCP_ZONE, config.GCP_MACHINE_TYPE, n_nodes
    )

    await gke_cluster.start()
    current_clusters[gke_cluster.cluster_id] = gke_cluster

    # Create deployment
    deployment_id, service_url = await gke_cluster.create_deployment(
        container=config.MAPPER_CONTAINER, num_replicas=n_nodes
    )

    return json({"cluster_id": gke_cluster.cluster_id})


@app.route("/stop_cluster", methods=["POST"])
async def stop_cluster(request):
    cluster_id = request.form["cluster_id"]

    gke_cluster = current_clusters[cluster_id]
    await gke_cluster.stop()
    del current_clusters[cluster_id]

    return text("", status=204)


@app.route("/start", methods=["POST"])
async def start(request):
    cluster_id = request.form["cluster_id"]
    n_mappers = request.form["n_mappers"]
    bucket = request.form["bucket"]
    paths = request.form["paths"]

    print("cluster_id", cluster_id)
    cluster = current_clusters[cluster_id]

    # Get template
    job = MapReduceJob(
        MapperSpec(url=service_url),
        EmbeddingDictReducer(config.EMBEDDING_LAYER),
        {"input_bucket": bucket},
        n_retries=N_RETRIES,
    )

    query_id = job.job_id
    current_queries[query_id] = job

    # Get list of paths
    iterable = paths

    cleanup_func = functools.partial(cleanup_query, query_id=query_id, dataset=iterable)

    await job.start(iterable, iterable.close)

    return json({"query_id": query_id})

    # dataset = FileListIterator(config.IMAGE_LIST_PATH)
    # cleanup_func = functools.partial(cleanup_query, query_id=query_id, dataset=dataset)
    # await query_job.start(dataset, cleanup_func)


@app.route("/results", methods=["GET"])
async def get_results(request):
    query_id = request.args["query_id"][0]
    query_job = current_queries[query_id]
    if query_job.finished:
        current_queries.pop(query_id)

    results = query_job.job_result
    results["result"] = [r.to_dict() for r in results["result"]]  # make serializable
    return json(results)


@app.route("/stop", methods=["PUT"])
async def stop(request):
    query_id = request.json["query_id"]
    await current_queries.pop(query_id).stop()
    return text("", status=204)


def cleanup_query(_, query_id: str, dataset: FileListIterator):
    dataset.close()
    asyncio.create_task(final_query_cleanup(query_id))


async def final_query_cleanup(query_id: str):
    await asyncio.sleep(config.QUERY_CLEANUP_TIME)
    current_queries.pop(query_id, None)  # don't throw error if already deleted


# INDEX MANAGEMENT


class LabeledIndex:
    def __init__(self, embedding_dict, **kwargs):
        self.labels = list(embedding_dict.keys())
        vectors = np.concatenate(list(embedding_dict.values()))

        self.index = InteractiveIndex(**kwargs)

        # Train
        # TODO(mihirg): Train only on the first few results and add incrementally
        # to index as more come back
        self.index.train(vectors)

        # Add
        # TODO(mihirg): Add incrementally as results come back
        self.index.add(vectors)

        self.index.merge_partial_indexes()

    def delete(self):
        self.index.cleanup()

    def query(self, query_vector, num_results, num_probes):
        dists, inds = self.index.query(query_vector, num_results, n_probes=num_probes)
        assert len(inds) == 1
        return [(self.labels[i], dist) for i, dist in zip(inds[0], dists[0])]


current_indexes = {}  # type: Dict[str, LabeledIndex]


@app.route("/create_index", methods=["POST"])
async def create_index(request):
    query_id = request.form["query_id"]
    embedding_dict = current_queries[query_id].result

    index_id = str(uuid.uuid4())
    current_indexes[index_id] = LabeledIndex(
        embedding_dict,
        d=config.EMBEDDING_DIM,
        n_centroids=config.INDEX_NUM_CENTROIDS,
        vectors_per_index=config.INDEX_SUBINDEX_SIZE,
    )
    return json({"index_id": index_id})


@app.route("/query_index", methods=["POST"])
async def query_index(request):
    embedding_endpoint = request.form["embedding_endpoint"]
    query_image_path = request.form["query_image_path"]
    index_id = request.form["index_id"]
    num_results = int(request.form["num_results"])

    # Generate query vector
    job = MapReduceJob(
        MapperSpec(url=embedding_endpoint),
        EmbeddingDictReducer(config.EMBEDDING_LAYER),
        {"input_bucket": config.IMAGE_BUCKET},
        n_retries=config.N_RETRIES,
    )
    query_vector_dict = await job.run_until_complete([query_image_path])
    query_vector = next(iter(query_vector_dict.values()))

    # Run query and return results
    query_results = current_indexes[index_id].query(
        query_vector, num_results, config.INDEX_NUM_QUERY_PROBES
    )
    return json(query_results)


@app.route("/delete_index", methods=["POST"])
async def delete_index(request):
    index_id = request.form["index_id"]
    current_indexes.pop(index_id).delete()
    return text("", status=204)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
