import psutil

INDEX_MAX_RAM_BYTES = psutil.virtual_memory().total * 3 // 4
INDEX_PCA_DIM = 128
INDEX_SQ_BYTES = 8

INDEX_SUBINDEX_SIZE = 25_000
INDEX_USE_GPU = False
INDEX_TRAIN_ON_GPU = True

INDEX_DIR_PATTERN = "/shared/indexes/{}"
ENVVAR_PREFIX = "FORAGER"
