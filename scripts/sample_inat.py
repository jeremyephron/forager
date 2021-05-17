import asyncio
from dataclasses import dataclass, field
import functools
import json
import os
import random

from typing import (
    Any,
    AsyncGenerator,
    AsyncIterable,
    Coroutine,
    List,
)

from gcloud.aio.storage import Storage
import click


def unasync(coro):
    @functools.wraps(coro)
    def wrapper(*args, **kwargs):
        return asyncio.run(coro(*args, **kwargs))

    return wrapper


@dataclass
class _LimitedAsCompletedState:
    pending: List[asyncio.Future] = field(default_factory=list)
    hit_stop_iteration = False
    next_coro_is_pending = False


async def limited_as_completed_from_async_coro_gen(
    coros: AsyncIterable[Coroutine[Any, Any, Any]], limit: int
) -> AsyncGenerator[asyncio.Task, None]:
    state = _LimitedAsCompletedState()
    NEXT_CORO_TASK_NAME = "get_next_coro"

    async def get_next_coro():
        try:
            coro = await coros.__anext__()
            return coro
        except StopAsyncIteration:
            state.hit_stop_iteration = True

    def schedule_getting_next_coro():
        task = asyncio.create_task(get_next_coro())
        task.get_name = lambda: NEXT_CORO_TASK_NAME  # patch for Python 3.7
        state.pending.append(task)
        state.next_coro_is_pending = True

    schedule_getting_next_coro()

    while state.pending:
        done_set, pending_set = await asyncio.wait(
            state.pending, return_when=asyncio.FIRST_COMPLETED
        )
        state.pending = list(pending_set)

        for done in done_set:
            assert isinstance(done, asyncio.Task)
            if done.get_name() == NEXT_CORO_TASK_NAME:
                state.next_coro_is_pending = False
                if state.hit_stop_iteration:
                    continue

                # Schedule the new coroutine
                state.pending.append(asyncio.create_task(done.result()))

                # If we have capacity, also ask for the next coroutine
                if len(state.pending) < limit:
                    schedule_getting_next_coro()
            else:
                # We definitely have capacity now, so ask for the next coroutine if
                # we haven't already
                if not state.next_coro_is_pending and not state.hit_stop_iteration:
                    schedule_getting_next_coro()

                yield done


def parse_gcs_path(path):
    assert path.startswith("gs://")
    path = path[len("gs://") :]
    bucket_end = path.find("/")
    bucket = path[:bucket_end]
    relative_path = path[bucket_end:].strip("/")
    return bucket, relative_path


async def process_image_gen(images, src_gcs_path, dst_gcs_path, sampling_rate, client):
    src_bucket, src_path = parse_gcs_path(src_gcs_path)
    dst_bucket, dst_path = parse_gcs_path(dst_gcs_path)

    for image in images:
        if random.random() > sampling_rate:
            continue
        filename = image["file_name"]
        yield client.copy(
            src_bucket,
            os.path.join(src_path, filename),
            dst_bucket,
            new_name=os.path.join(dst_path, os.path.basename(filename)),
        )


@click.command()
@click.argument("split_json", type=click.File("r"))
@click.argument("src_gcs_path")
@click.argument("dst_gcs_path")
@click.option("--sampling_rate", default=1.0, type=float)
@click.option("--limit", default=10, type=int)
@unasync
async def main(split_json, src_gcs_path, dst_gcs_path, sampling_rate, limit):
    images = json.load(split_json)["images"]

    async with Storage() as client:
        async for result in limited_as_completed_from_async_coro_gen(
            process_image_gen(
                images, src_gcs_path, dst_gcs_path, sampling_rate, client
            ),
            limit,
        ):
            await result


if __name__ == "__main__":
    main()
