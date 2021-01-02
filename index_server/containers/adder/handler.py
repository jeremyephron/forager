import asyncio
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

    def transform(
        self, embedding_dict: Dict[int, np.ndarray]
    ) -> Tuple[List[np.ndarray], List[int]]:
        all_embeddings = list(map(self.reduction, embedding_dict.values()))
        all_ids = [
            int(id)
            for id, embeddings in zip(embedding_dict, all_embeddings)
            for _ in range(embeddings.shape[0])
        ]
        return all_embeddings, all_ids

    def add(self, all_embeddings: List[np.ndarray], all_ids: List[int]):
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
        embedding_dict = await self.load(input, request_id)
        num_added = await asyncio.gather(
            *[
                self.build_index(index_type, index, embedding_dict, request_id)
                for index_type, index in indexes.items()
            ]
        )
        return dict(zip(indexes.keys(), num_added))

    async def load(self, path: str, request_id: str) -> Dict[int, np.ndarray]:
        # Step 1: Load saved embeddings into memory
        return await self.apply_in_executor(
            lambda p: np.load(p, allow_pickle=True).item(),
            path,
            request_id=request_id,
            profiler_name="load_time",
        )

    async def build_index(
        self,
        index_type: str,
        index: Index,
        embedding_dict: Dict[int, np.ndarray],
        request_id: str,
    ) -> int:
        # Step 2: Transform embeddings (perform any necessary reduction)
        all_embeddings, all_ids = await self.apply_in_executor(
            index.transform,
            embedding_dict,
            request_id=request_id,
            profiler_name=f"{index_type}_transform_time",
        )

        # Step 3: Add to on-disk index
        await self.apply_in_executor(
            index.add,
            all_embeddings,
            all_ids,
            request_id=request_id,
            profiler_name=f"{index_type}_add_time",
        )

        return len(all_ids)  # number of embeddings added to index

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
