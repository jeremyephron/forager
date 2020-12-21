from pathlib import Path
from interactive_index.utils import GIGABYTE

SANIC_RESPONSE_TIMEOUT = 10 * 60  # seconds

CLOUD_RUN_N_MAPPERS = 50
CLOUD_RUN_N_RETRIES = 1

CLUSTER_TERRAFORM_MODULE_PATH = Path("./terraform").resolve()
CLUSTER_REUSE_EXISTING = True
CLUSTER_MOUNT_DIR = Path("~/forager/mount").expanduser().resolve()

MAPPER_NUM_RETRIES = 5
MAPPER_CHUNK_SIZE = 5
MAPPER_REQUEST_TIMEOUT = 5 * 60  # seconds
MAPPER_CLOUD_RUN_URL = "https://forager-index-mapper-g6rwrca4fq-uw.a.run.app"

ADDER_NUM_RETRIES = 5
ADDER_CHUNK_SIZE = 5
ADDER_REQUEST_TIMEOUT = 10 * 60  # seconds

NUM_EMBEDDINGS_PER_IMAGE = 400
EMBEDDING_DIM = 2048
INDEX_PCA_DIM = 128
INDEX_SQ_BYTES = 8

# TODO(mihirg): Query GCP metadata for IP
TRAINING_MAX_RAM = 35 * GIGABYTE
TRAINER_N_CENTROIDS_MULTIPLE = 39
TRAINER_EMBEDDING_SAMPLE_RATE = 0.1
TRAINER_STATUS_ENDPOINT = "/trainer_status"
TRAINER_STATUS_CALLBACK = f"http://35.199.179.109:5000{TRAINER_STATUS_ENDPOINT}"

INDEX_PARENT_DIR = Path("~/forager/indexes").expanduser().resolve()
INDEX_UPLOAD_GCS_PATH = "gs://forager/indexes/"  # trailing slash = directory

QUERY_PATCHES_PER_IMAGE = 8
QUERY_NUM_RESULTS_MULTIPLE = 80
