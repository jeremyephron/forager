import abc
import asyncio
import collections
from dataclasses import dataclass
import functools
import time
import uuid

from typing import List, Dict, Any, DefaultDict

from sanic import Sanic
from sanic.response import json
from sanic_compress import Compress

from knn.utils import JSONType


@dataclass
class RequestProfiler:
    request_id: str
    category: str
    results_dict: Dict[str, DefaultDict[str, float]]
    additional: float = 0.0

    def __enter__(self):
        self.start_time = time.time()

    def __exit__(self, type, value, traceback):
        self.results_dict[self.request_id][self.category] += (
            time.time() - self.start_time + self.additional
        )


class Mapper(abc.ABC):
    # BASE CLASS

    def initialize_container(self, *args, **kwargs) -> None:
        pass

    async def initialize_job(self, job_args: JSONType) -> Any:
        return job_args

    async def process_chunk(
        self, chunk: List[JSONType], job_id: str, job_args: Any, request_id: str
    ) -> List[JSONType]:
        return await asyncio.gather(
            *[
                self.process_element(input, job_id, job_args, request_id, i)
                for i, input in enumerate(chunk)
            ]
        )

    @abc.abstractmethod
    async def process_element(
        self,
        input: JSONType,
        job_id: str,
        job_args: Any,
        request_id: str,
        element_index: int,
    ) -> JSONType:
        pass

    # DECORATORS

    @staticmethod
    def SkipIfAssertionError(f):
        @functools.wraps(f)
        async def wrapper(*args, **kwargs):
            try:
                return await f(*args, **kwargs)
            except AssertionError:
                pass
            return None

        return wrapper

    @staticmethod
    def SkipIfError(f):
        @functools.wraps(f)
        async def wrapper(*args, **kwargs):
            try:
                return await f(*args, **kwargs)
            except Exception:
                pass
            return None

        return wrapper

    # INTERNAL

    def __init__(self, *args, start_server=True, **kwargs):
        self._init_start_time = time.time()

        self.worker_id = str(uuid.uuid4())
        self._args_by_job: Dict[str, Any] = {}
        self._profiling_results_by_request: DefaultDict[
            str, DefaultDict[str, float]
        ] = collections.defaultdict(lambda: collections.defaultdict(float))
        self.profiler = functools.partial(
            RequestProfiler, results_dict=self._profiling_results_by_request
        )

        self.initialize_container(*args, **kwargs)

        if start_server:
            self._server = Sanic(self.worker_id)
            Compress(self._server)
            self._server.add_route(self._handle_request, "/", methods=["POST"])
            self._server.add_route(self._sleep, "/sleep", methods=["POST"])
        else:
            self._server = None

        self._init_time = time.time() - self._init_start_time

    async def __call__(self, *args, **kwargs):
        if self._server is not None:
            return await self._server(*args, **kwargs)

    async def _handle_request(self, request):
        init_time = self._init_time
        self._init_time = 0.0

        request_id = str(uuid.uuid4())
        with self.profiler(request_id, "billed_time", additional=init_time):
            with self.profiler(request_id, "request_time"):
                job_id = request.json["job_id"]
                job_args = self._args_by_job.setdefault(
                    job_id, await self.initialize_job(request.json["job_args"])
                )  # memoized
                outputs = await self.process_chunk(
                    request.json["inputs"], job_id, job_args, request_id
                )

        return json(
            {
                "worker_id": self.worker_id,
                "profiling": self._profiling_results_by_request.pop(request_id),
                "outputs": outputs,
            }
        )

    async def _sleep(self, request):
        delay = float(request.json["delay"])
        await asyncio.sleep(delay)
        return json(request.json)
