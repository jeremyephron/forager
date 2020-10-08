import asyncio
import base64
import functools
import io
import itertools

import numpy as np

from typing import Any, List, Union, Dict, Iterator

JSONType = Union[str, int, float, bool, None, Dict[str, Any], List[Any]]


class FileListIterator:
    def __init__(self, list_path: str, map_fn=lambda line: line) -> None:
        self.map_fn = map_fn

        self._list = open(list_path, "r")
        self._total = 0
        for line in self._list:
            if not line.strip():
                break
            self._total += 1
        self._list.seek(0)

    def close(self, *args, **kwargs):
        self._list.close()

    def __len__(self):
        return self._total

    def __iter__(self) -> Iterator[str]:
        return self

    def __next__(self) -> str:
        elem = self._list.readline().strip()
        if not elem:
            raise StopIteration
        return self.map_fn(elem)


def limited_as_completed(coros, limit):
    # Based on https://github.com/andybalaam/asyncioplus/blob/master/asyncioplus/limited_as_completed.py  # noqa
    futures = [asyncio.create_task(c) for c in itertools.islice(coros, 0, limit)]
    pending = [len(futures)]  # list so that we can modify from first_to_finish

    async def first_to_finish():
        while True:
            await asyncio.sleep(0)
            for i, f in enumerate(futures):
                if f is not None and f.done():
                    try:
                        newf = next(coros)
                        futures[i] = asyncio.create_task(newf)
                    except StopIteration:
                        futures[i] = None
                        pending[0] -= 1
                    return f.result()

    while pending[0] > 0:
        yield first_to_finish()


def unasync(f):
    @functools.wraps(f)
    def wrapper(*args, **kwargs):
        return asyncio.run(f(*args, **kwargs))

    return wrapper


def chunk(it, chunk_size, until=None):
    n = 0
    while True:
        this_chunk_size = chunk_size
        if until:
            if n >= until:
                break
            this_chunk_size = min(this_chunk_size, until - n)

        chunk = list(itertools.islice(it, this_chunk_size))
        if not chunk:
            break
        n += len(chunk)
        yield chunk


def numpy_to_base64(nda):
    with io.BytesIO() as nda_buffer:
        np.save(nda_buffer, nda, allow_pickle=False)
        nda_bytes = nda_buffer.getvalue()
        nda_base64 = base64.b64encode(nda_bytes).decode("ascii")
    return nda_base64


def base64_to_numpy(nda_base64):
    nda_bytes = base64.b64decode(nda_base64)
    with io.BytesIO(nda_bytes) as nda_buffer:
        nda = np.load(nda_buffer, allow_pickle=False)
    return nda
