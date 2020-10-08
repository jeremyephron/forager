import numpy as np
import asyncio
import functools
from operator import itemgetter

from typing import Dict

from jinja2 import Environment, FileSystemLoader, select_autoescape
from sanic import Sanic
from sanic.response import json, html, text

from knn.jobs import MapReduceJob
from knn.reducers import TopKReducer, PoolingReducer
from knn.utils import FileListIterator, numpy_to_base64
from knn.clusters import GKECluster

import config


# Start web server
app = Sanic(__name__)
app.static("/static", "./static")
app.update_config({
    'RESPONSE_TIMEOUT': 10 * 60 # 10 minutes
})
jinja = Environment(
    loader=FileSystemLoader("./templates"), autoescape=select_autoescape(["html"]),
)

current_clusters = {}  # type: Dict[str, GKECluster]
current_queries = {}  # type: Dict[str, MapReduceJob]
current_indices = {}  # type: Dict[str, InteractiveIndex]


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
        config.GCP_PROJECT,
        config.GCP_ZONE,
        config.GCP_MACHINE_TYPE,
        n_nodes)

    await gke_cluster.start()
    current_clusters[gke_cluster.cluster_id] = gke_cluster

    # Create deployment
    deployment_id, service_url = await gke_cluster.create_deployment(
        container=config.MAPPER_CONTAINER,
        num_replicas=n_nodes
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

    print('cluster_id', cluster_id)
    cluster = current_clusters[cluster_id]

    # Get template
    job = MapReduceJob(
        MapperSpec(url=service_url),
        EmbeddingDictReducer(),
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

    #dataset = FileListIterator(config.IMAGE_LIST_PATH)
    #cleanup_func = functools.partial(cleanup_query, query_id=query_id, dataset=dataset)
    #await query_job.start(dataset, cleanup_func)


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

# Index
@app.route("/create_index", methods=["POST"])
async def create_index(request):
    query_id = request.json["query_id"]
    #await current_queries.pop(query_id).stop()
    index_id = None
    return json({'index_id': index_id})


@app.route("/query_index")
async def query_index(request):
    index_id = request.json["index_id"]
    query_paths = request.json["query_paths"]
    num_results = request.json["num_results"]
    #await current_queries.pop(query_id).stop()
    query_results = {}
    return json(query_results)


@app.route("/delete_index")
async def delete_index(request, methods=["POST"]):
    index_id = request.json["index_id"]
    #await current_queries.pop(query_id).stop()
    return text("", status=204)


def cleanup_query(_, query_id: str, dataset: FileListIterator):
    dataset.close()
    asyncio.create_task(final_query_cleanup(query_id))


async def final_query_cleanup(query_id: str):
    await asyncio.sleep(config.QUERY_CLEANUP_TIME)
    current_queries.pop(query_id, None)  # don't throw error if already deleted


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
