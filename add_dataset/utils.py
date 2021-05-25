import asyncio
import functools
import os


def parse_gcs_path(path):
    assert path.startswith("gs://")
    path = path[len("gs://") :]
    bucket_end = path.find("/")
    bucket = path[:bucket_end]
    relative_path = path[bucket_end:].strip("/")
    return bucket, relative_path


def make_identifier(path):
    return os.path.splitext(os.path.basename(path))[0]


def unasync(coro):
    @functools.wraps(coro)
    def wrapper(*args, **kwargs):
        return asyncio.run(coro(*args, **kwargs))

    return wrapper
