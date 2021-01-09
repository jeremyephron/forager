import os

SHARD_INDEX_NAME_TMPL = "shard_{}_{{}}.index"

NPROC = int(os.getenv("NPROC", "1"))
