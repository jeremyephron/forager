GCP_PROJECT = "visualdb-1046"
GCP_ZONE = "us-central1-a"
GCP_MACHINE_TYPE = "n2-highcpu-4"

OUTPUT_FILENAME = "lvis-embeddings.npy"
MAPPER_CONTAINER = "gcr.io/visualdb-1046/mihir-demo-full-image-embedding"

QUERY_CLEANUP_TIME = 60 * 60  # 1 hour
N_RETRIES = 100

UPDATE_INTERVAL = 1

EMBEDDING_LAYER = "res5"
EMBEDDING_DIM = 2048
INDEX_NUM_CENTROIDS = 1024
INDEX_SUBINDEX_SIZE = 25000
INDEX_NUM_QUERY_PROBES = 4
INDEX_USE_GPU = False
