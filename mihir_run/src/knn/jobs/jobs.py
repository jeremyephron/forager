import asyncio
import collections
import itertools
import resource
import time
import uuid

import aiohttp
from runstats import Statistics

from knn import utils
from knn.utils import JSONType
from knn.reducers import Reducer

from . import defaults

from typing import (
    Optional,
    Callable,
    Tuple,
    List,
    Dict,
    Any,
    Iterable,
)


# Increase maximum number of open sockets if necessary
soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
new_soft = max(min(defaults.DESIRED_ULIMIT, hard), soft)
resource.setrlimit(resource.RLIMIT_NOFILE, (new_soft, hard))


class MapReduceJob:
    def __init__(
        self,
        mapper_url: str,
        reducer: Reducer,
        mapper_args: JSONType = {},
        *,
        n_mappers: int = defaults.N_MAPPERS,
        n_retries: int = defaults.N_RETRIES,
        chunk_size: int = defaults.CHUNK_SIZE,
    ) -> None:
        assert n_mappers < new_soft

        self.job_id = str(uuid.uuid4())

        self.n_mappers = n_mappers
        self.n_retries = n_retries
        self.chunk_size = chunk_size
        self.mapper_url = mapper_url
        self.mapper_args = mapper_args

        self.reducer = reducer

        # Performance stats
        self._n_requests = 0
        self._n_successful = 0
        self._n_failed = 0
        self._n_chunks_per_mapper: Dict[str, int] = collections.defaultdict(int)
        self._profiling: Dict[str, Statistics] = collections.defaultdict(Statistics)

        # Will be initialized later
        self._n_total: Optional[int] = None
        self._start_time: Optional[float] = None
        self._task: Optional[asyncio.Task] = None

    # REQUEST LIFECYCLE

    async def start(
        self,
        iterable: Iterable[JSONType],
        callback: Optional[Callable[[Dict[str, Any]], None]] = None,
    ) -> None:
        async def task():
            try:
                result = await self.run_until_complete(iterable)
            except asyncio.CancelledError:
                pass
            else:
                if callback is not None:
                    callback(result)

        self.task = asyncio.create_task(task())

    async def run_until_complete(self, iterable: Iterable[JSONType]) -> Dict[str, Any]:
        assert self._start_time is None  # can't reuse Job instances
        self.start_time = time.time()

        try:
            self._n_total = len(iterable)
        except Exception:
            pass

        iterable = iter(iterable)

        if self.chunk_size == 15:
            print("Using chunk size schedule 1 -> 5")
            chunked = itertools.chain(
                utils.chunk(iterable, 1, until=2 * self.n_mappers),
                utils.chunk(iterable, 5),
            )
        elif self.chunk_size == 135:
            print("Using chunk size schedule 1 -> 3 -> 5")
            chunked = itertools.chain(
                utils.chunk(iterable, 1, until=2 * self.n_mappers),
                utils.chunk(iterable, 3, until=2 * self.n_mappers),
                utils.chunk(iterable, 5),
            )
        else:
            chunked = utils.chunk(iterable, self.chunk_size)

        connector = aiohttp.TCPConnector(limit=0)
        async with aiohttp.ClientSession(connector=connector) as session:
            for response_tuple in utils.limited_as_completed(
                (self._request(session, chunk) for chunk in chunked), self.n_mappers,
            ):
                self._handle_chunk_result(*(await response_tuple))

        if self._n_total is None:
            self._n_total = self._n_successful + self._n_failed
        else:
            assert self._n_total == self._n_successful + self._n_failed

        return self.result

    async def stop(self) -> None:
        if self.task is not None and not self.task.done():
            self.task.cancel()
            await self.task

    # RESULT GETTERS

    @property
    def result(self) -> Any:
        return self.reducer.result

    @property
    def job_result(self) -> Dict[str, Any]:
        elapsed_time = (time.time() - self.start_time) if self.start_time else 0.0

        performance = {
            "profiling": {k: v.mean() for k, v in self._profiling.items()},
            "mapper_utilization": dict(enumerate(self._n_chunks_per_mapper.values())),
        }

        progress = {
            "cost": self.cost,
            "finished": self.finished,
            "n_processed": self._n_successful,
            "n_skipped": self._n_failed,
            "elapsed_time": elapsed_time,
        }
        if self._n_total is not None:
            progress["n_total"] = self._n_total

        return {
            "performance": performance,
            "progress": progress,
            "result": self.result,
        }

    @property
    def finished(self) -> bool:
        return self._n_total == self._n_successful + self._n_failed

    @property
    def cost(self) -> float:
        total_billed_time = self._profiling["billed_time"].mean() * len(
            self._profiling["billed_time"]
        )
        return (
            0.00002400 * total_billed_time
            + 2 * 0.00000250 * total_billed_time
            + 0.40 / 1000000 * self._n_requests
        )

    # INTERNAL

    def _construct_request(self, chunk: List[JSONType]) -> JSONType:
        return {
            "job_id": self.job_id,
            "job_args": self.mapper_args,
            "inputs": chunk,
        }

    async def _request(
        self, session: aiohttp.ClientSession, chunk: List[JSONType]
    ) -> Tuple[JSONType, Optional[JSONType], float]:
        result = None
        start_time = 0.0
        end_time = 0.0

        request = self._construct_request(chunk)

        for i in range(self.n_retries):
            start_time = time.time()
            end_time = start_time

            try:
                async with session.post(self.mapper_url, json=request) as response:
                    end_time = time.time()
                    if response.status == 200:
                        result = await response.json()
                        break
            except aiohttp.ClientConnectionError:
                break

        return chunk, result, end_time - start_time

    def _handle_chunk_result(
        self, chunk: List[JSONType], result: Optional[JSONType], elapsed_time: float
    ):
        self._n_requests += 1

        if not result:
            self._n_failed += len(chunk)
            return

        # Validate
        assert len(result["outputs"]) == len(chunk)
        assert "billed_time" in result["profiling"]

        n_successful = sum(1 for r in result["outputs"] if r)
        self._n_successful += n_successful
        self._n_failed += len(chunk) - n_successful
        self._n_chunks_per_mapper[result["worker_id"]] += 1

        self._profiling["total_time"].push(elapsed_time)
        for k, v in result["profiling"].items():
            self._profiling[k].push(v)

        for input, output in zip(chunk, result["outputs"]):
            if output:
                self.reducer.handle_result(input, output)
