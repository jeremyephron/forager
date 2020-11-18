GCP_PROJECT = "visualdb-1046"
GCP_ZONE = "us-west1-b"
GCP_MACHINE_TYPE = "n2-highcpu-4"

OUTPUT_FILENAME = "lvis-embeddings.npy"
MAPPER_CLOUD_RUN_URL = (
    "https://vishnu-demo-full-image-embedding-spatial-g6rwrca4fq-uw.a.run.app"
)
MAPPER_CONTAINER = "gcr.io/visualdb-1046/vishnu-demo-full-image-embedding-spatial"
CLUSTER_NODE_N_MAPPERS = 2
CLOUD_RUN_N_MAPPERS = 70

N_RETRIES = 1
CHUNK_SIZE = 1

EMBEDDING_LAYER = "res5"
EMBEDDING_DIM = 2048

QUERY_PATCHES_PER_IMAGE = 8
QUERY_NUM_RESULTS_MULTIPLE = 80

INDEX_NUM_CENTROIDS = 64
INDEX_SUBINDEX_SIZE = 5_000_000
INDEX_NUM_QUERY_PROBES = 16
INDEX_USE_GPU = False
INDEX_TRAIN_MULTIPLE = 39

INDEX_TRANSFORM = "PCA"
INDEX_TRANSFORM_ARGS = [64]
INDEX_ENCODING = "SQ"
INDEX_ENCODING_ARGS = [8]

INDEX_FLUSH_SLEEP = 1  # seconds

INDEX_UPLOAD_GCS_PATH = "gs://forager/indexes/"  # trailing slash = directory

CLEANUP_TIMEOUT = 30 * 60  # seconds
CLEANUP_INTERVAL = CLEANUP_TIMEOUT // 4
