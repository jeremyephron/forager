import os.path
from pathlib import Path
from interactive_index.utils import GIGABYTE
from urllib.request import Request, urlopen

# Get instance IP (https://stackoverflow.com/q/23362887)
ip_request = Request(
    "http://metadata/computeMetadata/v1/instance/network-interfaces/0/access-configs/0/external-ip"
)
ip_request.add_header("Metadata-Flavor", "Google")
INSTANCE_IP = urlopen(ip_request).read().decode()
# INSTANCE_IP = ""

SANIC_RESPONSE_TIMEOUT = 10 * 60  # seconds

CLOUD_RUN_N_MAPPERS = 50
CLOUD_RUN_N_RETRIES = 1

CLUSTER_TERRAFORM_MODULE_PATH = Path("./terraform").resolve()
CLUSTER_REUSE_EXISTING = True
CLUSTER_MOUNT_DIR = Path("~/forager/mount").expanduser().resolve()
CLUSTER_CLEANUP_TIME = 20 * 60  # seconds (destroy cluster after idle for this long)

MAPPER_NUM_RETRIES = 5
MAPPER_CHUNK_SIZE = lambda nproc: 3
MAPPER_REQUEST_MULTIPLE = lambda nproc: nproc
MAPPER_REQUEST_TIMEOUT = (
    3 * 60
)  # seconds; more than a minute per image is probably too much
MAPPER_CLOUD_RUN_URL = "https://forager-index-mapper-g6rwrca4fq-uc.a.run.app"

LOCAL_INDEX_BUILDING_NUM_THREADS = 10

ADDER_NUM_RETRIES = 5
ADDER_CHUNK_SIZE = lambda nproc: 1
ADDER_REQUEST_MULTIPLE = lambda nproc: nproc
ADDER_REQUEST_TIMEOUT = 5 * 60  # seconds

RESIZER_NUM_RETRIES = 20  # all should succeed
RESIZER_CHUNK_SIZE = lambda nproc: 20
RESIZER_REQUEST_MULTIPLE = lambda nproc: nproc
RESIZER_REQUEST_TIMEOUT = 60  # seconds
RESIZER_OUTPUT_BUCKET = "foragerml"
RESIZER_OUTPUT_DIR_TMPL = "thumbnails/{}/"
RESIZER_MAX_HEIGHT = 200

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

BGSPLIT_TRAINING_MAX_RAM = 35 * GIGABYTE
BGSPLIT_TRAINER_STATUS_ENDPOINT = "/bgsplit_trainer_status"
BGSPLIT_TRAINER_STATUS_CALLBACK = (
    f"http://{INSTANCE_IP}:5000{BGSPLIT_TRAINER_STATUS_ENDPOINT}"
)

BGSPLIT_MAPPER_NUM_RETRIES = 5
BGSPLIT_MAPPER_CHUNK_SIZE = lambda nproc: 32
BGSPLIT_MAPPER_REQUEST_MULTIPLE = lambda nproc: nproc
BGSPLIT_MAPPER_REQUEST_TIMEOUT = (
    3 * 60
)  # seconds; more than a minute per image is probably too much
BGSPLIT_MAPPER_CLOUD_RUN_URL = "https://forager-bgsplit-mapper-g6rwrca4fq-uc.a.run.app"
BGSPLIT_MAPPER_JOB_DIR_TMPL = "shared/dnn_outputs/{}"

GCS_PUBLIC_ROOT_URL = "https://storage.googleapis.com/"

INDEX_PARENT_DIR = Path("~/forager/indexes").expanduser().resolve()
INDEX_UPLOAD_GCS_PATH = "gs://foragerml/indexes/"  # trailing slash = directory

MODEL_PARENT_DIR = Path("~/forager/models").expanduser().resolve()
MODEL_DIR_TMPL = Path("~/forager/models").expanduser().resolve()
MODEL_UPLOAD_GCS_PATH = "gs://foragerml/models/"  # trailing slash = directory

AUX_PARENT_DIR = Path("~/forager/aux_labels").expanduser().resolve()
AUX_DIR_TMPL = os.path.join(AUX_PARENT_DIR, "{}/{}.pickle")
AUX_UPLOAD_GCS_PATH = "gs://foragerml/aux_labels/"  # trailing slash = directory
AUX_GCS_TMPL = os.path.join(AUX_UPLOAD_GCS_PATH, "{}/{}.pickle")
AUX_GCS_PUBLIC_TMPL = os.path.join(
    GCS_PUBLIC_ROOT_URL, "foragerml/aux_labels/{}/{}.pickle"
)

MODEL_OUTPUTS_PARENT_DIR = Path("~/forager/model_outputs").expanduser().resolve()
MODEL_OUTPUTS_UPLOAD_GCS_PATH = (
    "gs://foragerml/models_outputs/"  # trailing slash = directory
)

QUERY_PATCHES_PER_IMAGE = 8
QUERY_NUM_RESULTS_MULTIPLE = 80

UPLOADED_IMAGE_BUCKET = "foragerml"
UPLOADED_IMAGE_DIR = "uploads/"

CLIP_TEXT_INFERENCE_CLOUD_RUN_URL = (
    "https://forager-clip-text-inference-g6rwrca4fq-uc.a.run.app"
)

BRUTE_FORCE_QUERY_CHUNK_SIZE = 512
DEFAULT_QUERY_MODEL = "imagenet"
EMBEDDING_DIMS_BY_MODEL = {
    "imagenet": 2048,
    "imagenet_early": 1024,
    "imagenet_full": 2048,
    "imagenet_full_early": 1024,
    "clip": 512,
}
BGSPLIT_EMBEDDING_DIM = 2048

EMBEDDING_FILE_NAME = "embeddings.npy"
MODEL_SCORES_FILE_NAME = "scores.npy"

DNN_SCORE_CLASSIFICATION_THRESHOLD = 0.5
ACTIVE_VAL_STARTING_BUDGET = 10

MIN_TIME_BETWEEN_KEEP_ALIVES = 10  # seconds
