import concurrent
import functools
from queue import SimpleQueue

import numpy as np

from typing import Any, Dict, List, Tuple

from interactive_index import InteractiveIndex

from knn import utils
from knn.mappers import Mapper
from knn.utils import JSONType

import config


TransformResult = Tuple[List[np.ndarray], List[int]]


class Index:
    def __init__(self, index_dict: Dict[str, Any], worker_id: str):
        average: bool = index_dict.get("average", False)
        index_dir: str = index_dict["index_dir"]

        if average:
            self.reduction = functools.partial(np.mean, axis=0, keepdims=True)
        else:
            self.reduction = lambda x: x

        self.indexes: SimpleQueue[InteractiveIndex] = SimpleQueue()
        for i in range(config.NPROC):
            index = InteractiveIndex.load(index_dir)
            index.SHARD_INDEX_NAME_TMPL = config.SHARD_INDEX_NAME_TMPL.format(
                worker_id, i
            )
            self.indexes.put(index)

    def transform(self, embedding_dict: Dict[int, np.ndarray]) -> TransformResult:
        all_embeddings = list(map(self.reduction, embedding_dict.values()))
        all_ids = [
            int(id)
            for id, embeddings in zip(embedding_dict, all_embeddings)
            for _ in range(embeddings.shape[0])
        ]
        return all_embeddings, all_ids

    def add(self, transform_result: TransformResult):
        all_embeddings, all_ids = transform_result
        index = self.indexes.get()  # get an index no other thread is adding to
        index.add(np.concatenate(all_embeddings), all_ids, update_metadata=False)
        self.indexes.put(index)


class IndexBuildingMapper(Mapper):
    def initialize_container(self):
        self.shard_tmpl_for_glob = config.SHARD_INDEX_NAME_TMPL.format(
            self.worker_id, "*"
        ).format("*")

    def register_executor(self):
        return concurrent.futures.ThreadPoolExecutor() if config.NPROC > 1 else None

    async def initialize_job(self, job_args) -> InteractiveIndex:
        job_args["indexes"] = {
            index_type: Index(index_dict, self.worker_id)
            for index_type, index_dict in job_args["indexes"].items()
        }
        return job_args

    # input = path to a np.save'd Dict[int, np.ndarray] where each value is N x D
    @utils.log_exception_from_coro_but_return_none
    async def process_element(
        self, input, job_id, job_args, request_id, element_index
    ) -> Dict[str, int]:
        indexes = job_args["indexes"]

        embedding_dict = await self.apply_in_executor(
            self.load, input, request_id=request_id, profiler_name="load_time"
        )
        transform_results = await self.apply_in_executor(
            self.transform,
            indexes,
            embedding_dict,
            request_id=request_id,
            profiler_name="transform_time",
        )
        await self.apply_in_executor(
            self.add,
            indexes,
            transform_results,
            request_id=request_id,
            profiler_name="add_time",
        )

        return {
            index_type: len(all_ids)
            for index_type, (_, all_ids) in transform_results.items()
        }

    def load(self, path: str) -> Dict[int, np.ndarray]:
        # Step 1: Load saved embeddings into memory
        return np.load(input, allow_pickle=True).item()

    def transform(
        self, indexes: Dict[str, Index], embedding_dict: Dict[int, np.ndarray]
    ) -> Dict[str, TransformResult]:
        # Step 2: Transform embeddings (perform any necessary reductions)
        return {
            index_type: index.transform(embedding_dict)
            for index_type, index in indexes.items()
        }

    def add(
        self, indexes: Dict[str, Index], transform_results: Dict[str, TransformResult]
    ):
        # Step 3: Add to on-disk indexes
        for index_type, transform_result in transform_results.items():
            indexes[index_type].add(transform_result)

    async def postprocess_chunk(
        self,
        inputs,
        outputs,
        job_id,
        job_args,
        request_id,
    ) -> Tuple[str, List[JSONType]]:
        return self.shard_tmpl_for_glob, outputs


mapper = IndexBuildingMapper()
