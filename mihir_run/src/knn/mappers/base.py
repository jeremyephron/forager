from __future__ import annotations

import abc
import asyncio
import concurrent
import collections
from dataclasses import dataclass
import functools
import time
import uuid

from sanic import Sanic
from sanic.response import json
from sanic_compress import Compress

from typing import Any, DefaultDict, Dict, List, Optional, Tuple

from knn.utils import JSONType


@dataclass
class RequestProfiler:
    request_id: str
    category: str
    results_dict: RequestProfiler.ProfilingDictType
    additional: float = 0.0

    ProfilingDictType = DefaultDict[str, DefaultDict[str, List[float]]]

    def __enter__(self):
        self.start_time = time.perf_counter()

    def __exit__(self, type, value, traceback):
        self.results_dict[self.request_id][self.category].append(
            time.perf_counter() - self.start_time + self.additional
        )


class Mapper(abc.ABC):
    # BASE CLASS

    def initialize_container(self, *args, **kwargs) -> None:
        pass

    def register_executor(self) -> Optional[concurrent.futures.Executor]:
        return None

    async def initialize_job(self, job_args: JSONType) -> Any:
        return job_args

    async def process_chunk(
        self, chunk: List[JSONType], job_id: str, job_args: Any, request_id: str
    ) -> Tuple[JSONType, List[JSONType]]:
        return await asyncio.gather(
            *[
                self.process_element(input, job_id, job_args, request_id, i)
                for i, input in enumerate(chunk)
            ]
        )

    async def postprocess_chunk(
        self,
        inputs: List[JSONType],
        outputs: List[Any],
        job_id: str,
        job_args: Any,
        request_id: str,
    ) -> Tuple[JSONType, List[JSONType]]:
        return (
            None,
            outputs,
        )  # if this is not overridden, process_element must return JSONType

    @abc.abstractmethod
    async def process_element(
        self,
        input: JSONType,
        job_id: str,
        job_args: Any,
        request_id: str,
        element_index: int,
    ) -> Any:
        pass

    # UTILITY FUNCTIONS

    async def apply_in_executor(self, f, *args, request_id, profiler_name, **kwargs):
        def wrapper():
            start_time = time.perf_counter()
            result = f(*args, **kwargs)
            end_time = time.perf_counter()
            return result, end_time - start_time

        with self.profiler(request_id, f"{profiler_name}_total"):
            if self.executor:
                loop = asyncio.get_running_loop()
                result, exec_time = await loop.run_in_executor(
                    self.executor,
                    wrapper,
                )
            else:
                result, exec_time = wrapper()

        with self.profiler(request_id, profiler_name, additional=exec_time):
            pass

        return result

    # INTERNAL

    def __init__(self, *args, start_server=True, **kwargs):
        self._init_start_time = time.perf_counter()

        self.worker_id = str(uuid.uuid4())
        self._args_by_job: Dict[str, Any] = {}
        self._profiling_results_by_request: RequestProfiler.ProfilingDictType = (
            collections.defaultdict(lambda: collections.defaultdict(list))
        )
        self.profiler = functools.partial(
            RequestProfiler, results_dict=self._profiling_results_by_request
        )

        self.initialize_container(*args, **kwargs)
        self.executor = self.register_executor()

        if start_server:
            self._server = Sanic(self.worker_id)
            Compress(self._server)
            self._server.add_route(self._handle_request, "/", methods=["POST"])
            self._server.add_route(self._sleep, "/sleep", methods=["POST"])
        else:
            self._server = None

        self._init_time = time.perf_counter() - self._init_start_time

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

                chunk = request.json["inputs"]
                raw_outputs = await self.process_chunk(
                    chunk, job_id, job_args, request_id
                )
                chunk_output, final_outputs = await self.postprocess_chunk(
                    chunk, raw_outputs, job_id, job_args, request_id
                )

        return json(
            {
                "worker_id": self.worker_id,
                "profiling": self._profiling_results_by_request.pop(request_id),
                "outputs": final_outputs,
                "chunk_output": chunk_output,
            }
        )

    async def _sleep(self, request):
        delay = float(request.json["delay"])
        await asyncio.sleep(delay)
        return json(request.json)
