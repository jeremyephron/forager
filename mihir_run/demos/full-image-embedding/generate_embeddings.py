import asyncio

import numpy as np
from tqdm import tqdm

from knn import utils
from knn.clusters import GKECluster
from knn.jobs import MapReduceJob, MapperSpec
from knn.utils import FileListIterator
from knn.reducers import Reducer


GCP_PROJECT = "visualdb-1046"
GCP_ZONE = "us-central1-a"
GCP_MACHINE_TYPE = "n2-highcpu-2"

INPUT_BUCKET = "mihir-fast-queries"
IMAGE_LIST_PATH = "images.txt"
OUTPUT_FILENAME = "lvis-embeddings.npy"
MAPPER_CONTAINER = "gcr.io/visualdb-1046/mihir-demo-full-image-embedding"
N_MAPPERS = 50
N_NODES = N_MAPPERS  # TODO: figure out how to put 2 mappers on each node
N_RETRIES = 10

UPDATE_INTERVAL = 1


class EmbeddingDictReducer(Reducer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.embeddings = {}

    def handle_result(self, input, output):
        self.embeddings[input] = {
            k: utils.base64_to_numpy(v) for k, v in output.items()
        }

    @property
    def result(self):
        return self.embeddings


@utils.unasync
async def main():
    async with GKECluster(GCP_PROJECT, GCP_ZONE, GCP_MACHINE_TYPE, N_NODES) as cluster:
        job = MapReduceJob(
            MapperSpec(
                container=MAPPER_CONTAINER, cluster=cluster, n_mappers=N_MAPPERS
            ),
            EmbeddingDictReducer(),
            {
                "input_bucket": INPUT_BUCKET,
            },
            n_retries=N_RETRIES,
        )
        iterable = FileListIterator(IMAGE_LIST_PATH, lambda i: {"image": i})

        await job.start(iterable, iterable.close)
        with tqdm(total=len(iterable)) as pbar:
            while not job.finished:
                await asyncio.sleep(UPDATE_INTERVAL)
                progress = job.job_result["progress"]
                pbar.update(progress["n_processed"])

        np.save(OUTPUT_FILENAME, job.result)


if __name__ == "__main__":
    main()
