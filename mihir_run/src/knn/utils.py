import asyncio
import base64
import functools
import io
import textwrap
import traceback

import numpy as np

from typing import Any, Awaitable, AsyncIterable, List, Union, Dict, Iterator

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


async def limited_as_completed(coros: AsyncIterable[Awaitable[Any]], limit: int):
    pending: List[asyncio.Future] = []
    hit_stop_iteration = False
    next_coro_is_pending = False

    async def get_next_coro():
        global hit_stop_iteration
        try:
            return await coros.__anext__()
        except StopAsyncIteration:
            hit_stop_iteration = True

    def schedule_getting_next_coro():
        global next_coro_is_pending
        task = asyncio.create_task(get_next_coro())
        task.is_to_get_next_coro = True
        pending.append(task)
        next_coro_is_pending = True

    schedule_getting_next_coro()

    while pending:
        done_set, pending_set = await asyncio.wait(
            pending, return_when=asyncio.FIRST_COMPLETED
        )
        pending = list(pending_set)

        for done in done_set:
            if getattr(done, "is_to_get_next_coro", False):
                next_coro_is_pending = False
                if hit_stop_iteration:
                    continue

                # Schedule the new coroutine
                pending.append(asyncio.create_task(done.result()))

                # If we have capacity, also ask for the next coroutine
                if len(pending) < limit:
                    schedule_getting_next_coro()
            else:
                # We definitely have capacity now, so ask for the next coroutine if we
                # haven't already
                if not next_coro_is_pending and not hit_stop_iteration:
                    schedule_getting_next_coro()

                yield done.result()


def unasync(coro):
    @functools.wraps(coro)
    def wrapper(*args, **kwargs):
        return asyncio.run(coro(*args, **kwargs))

    return wrapper


def unasync_as_task(coro):
    @functools.wraps(coro)
    def wrapper(*args, **kwargs):
        return asyncio.create_task(coro(*args, **kwargs))

    return wrapper


def log_exception_from_coro_but_return_none(coro):
    @functools.wraps(coro)
    async def wrapper(*args, **kwargs):
        try:
            return await coro(*args, **kwargs)
        except Exception:
            print(f"Error from {coro.__name__}")
            print(textwrap.indent(traceback.format_exc(), "  "))
        return None

    return wrapper


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
