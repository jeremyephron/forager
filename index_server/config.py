from pathlib import Path
from interactive_index.utils import GIGABYTE
from urllib.request import Request, urlopen

# Get instance IP (https://stackoverflow.com/q/23362887)
ip_request = Request(
    "http://metadata/computeMetadata/v1/instance/network-interfaces/0/access-configs/0/external-ip"
)
ip_request.add_header("Metadata-Flavor", "Google")
INSTANCE_IP = urlopen(ip_request).read().decode()

SANIC_RESPONSE_TIMEOUT = 10 * 60  # seconds

CLOUD_RUN_N_MAPPERS = 200
CLOUD_RUN_N_RETRIES = 3

CLUSTER_TERRAFORM_MODULE_PATH = Path("./terraform").resolve()
CLUSTER_REUSE_EXISTING = False
CLUSTER_MOUNT_DIR = Path("~/forager/mount").expanduser().resolve()
CLUSTER_CLEANUP_TIME = 20 * 60  # seconds (destroy cluster after idle for this long)

MAPPER_NUM_RETRIES = 5
MAPPER_CHUNK_SIZE = lambda nproc: 3
MAPPER_REQUEST_MULTIPLE = lambda nproc: nproc
MAPPER_REQUEST_TIMEOUT = (
    3 * 60
)  # seconds; more than a minute per image is probably too much
MAPPER_CLOUD_RUN_URL = "https://forager-index-mapper-g6rwrca4fq-uc.a.run.app"

ADDER_NUM_RETRIES = 5
ADDER_CHUNK_SIZE = lambda nproc: 1
ADDER_REQUEST_MULTIPLE = lambda nproc: nproc
ADDER_REQUEST_TIMEOUT = 5 * 60  # seconds

NUM_IMAGES_TO_MAP_BEFORE_CONFIGURING_INDEX = 100
EMBEDDING_DIM = 2048
INDEX_PCA_DIM = 128
INDEX_SQ_BYTES = 8
INDEX_SUBINDEX_SIZE = 1_000_000

# TODO(mihirg): Toggle when/if FAISS supports merging direct map indexes
BUILD_UNCOMPRESSED_FULL_IMAGE_INDEX = False

TRAINING_MAX_RAM = 35 * GIGABYTE
TRAINER_N_CENTROIDS_MULTIPLE = 39
TRAINER_EMBEDDING_SAMPLE_RATE = 0.1
TRAINER_STATUS_ENDPOINT = "/trainer_status"
TRAINER_STATUS_CALLBACK = f"http://{INSTANCE_IP}:5000{TRAINER_STATUS_ENDPOINT}"

INDEX_PARENT_DIR = Path("~/forager/indexes").expanduser().resolve()
INDEX_UPLOAD_GCS_PATH = "gs://foragerml/indexes/"  # trailing slash = directory

QUERY_PATCHES_PER_IMAGE = 8
QUERY_NUM_RESULTS_MULTIPLE = 80
