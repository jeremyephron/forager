GCP_PROJECT = 'visualdb-1046'
GCP_ZONE = 'us-central1-a'
GCP_MACHINE_TYPE = 'n2-highcpu-2'

OUTPUT_FILENAME = "lvis-embeddings.npy"
MAPPER_CONTAINER = "gcr.io/visualdb-1046/mihir-demo-full-image-embedding"

QUERY_CLEANUP_TIME = 60 * 60 # 1 hour
N_MAPPERS = 1000
N_RETRIES = 100

UPDATE_INTERVAL = 1
