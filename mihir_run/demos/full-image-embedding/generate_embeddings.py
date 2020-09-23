import asyncio
import numpy as np

from knn import utils
from knn.jobs import MapReduceJob
from knn.utils import FileListIterator
from knn.reducers import Reducer


INPUT_BUCKET = "mihir-fast-queries"
IMAGE_LIST_PATH = "images.txt"
OUTPUT_FILENAME = "lvis-embeddings.npy"
MAPPER_ENDPOINT = "https://mihir-demo-full-image-embedding-g6rwrca4fq-uc.a.run.app"
N_MAPPERS = 1000
N_RETRIES = 100


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


job = MapReduceJob(
    MAPPER_ENDPOINT,
    EmbeddingDictReducer(),
    {
        "input_bucket": INPUT_BUCKET,
    },
    n_mappers=N_MAPPERS,
    n_retries=N_RETRIES,
)
results = asyncio.run(job.run_until_complete(FileListIterator(IMAGE_LIST_PATH)))

np.save(OUTPUT_FILENAME, results)
