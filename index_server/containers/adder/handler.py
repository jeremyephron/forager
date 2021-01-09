import asyncio
from collections import defaultdict
import concurrent
import itertools

import numpy as np

from typing import Dict, List, Optional, Tuple

from interactive_index import InteractiveIndex

from knn import utils
from knn.mappers import Mapper
from knn.utils import JSONType

import config


class Index:
    def __init__(self, index_dir: str, worker_id: str):
        self.index = InteractiveIndex.load(index_dir)
        self.index.SHARD_INDEX_NAME_TMPL = config.SHARD_INDEX_NAME_TMPL.format(
            worker_id
        )

    def add(self, embedding_dicts: List[Dict[int, np.ndarray]]):
        ids = [
            int(id)
            for embedding_dict in embedding_dicts
            for id, embeddings in embedding_dict.items()
            for _ in range(embeddings.shape[0])
        ]
        embeddings = np.concatenate(
            list(
                itertools.chain(
                    *[embedding_dict.values() for embedding_dict in embedding_dicts]
                )
            )
        )
        self.index.add(embeddings, ids, update_metadata=False)


class IndexBuildingMapper(Mapper):
    def initialize_container(self):
        self.shard_pattern_for_glob = config.SHARD_INDEX_NAME_TMPL.format(
            self.worker_id
        ).format("*")

    def register_executor(self):
        return concurrent.futures.ThreadPoolExecutor()

    async def initialize_job(self, job_args) -> InteractiveIndex:
        index_dicts = job_args["indexes"]

        job_args["indexes_by_reduction"] = defaultdict(dict)
        for index_name, index_dict in index_dicts.items():
            reduction = index_dict["reduction"]
            index_dir = index_dict["index_dir"]

            job_args["indexes_by_reduction"][reduction][index_name] = Index(
                index_dir, self.worker_id
            )

        return job_args

    # inputs = paths to np.save'd Dict[int, np.ndarray] where each value is N x D
    @utils.log_exception_from_coro_but_return_none
    async def process_chunk(
        self, chunk: List[str], job_id, job_args, request_id
    ) -> Optional[bool]:
        indexes_by_reduction = job_args["indexes_by_reduction"]
        path_tmpls = chunk

        for reduction, indexes in indexes_by_reduction.items():
            # Step 1: Load saved embeddings of entire chunk into memory
            with self.profiler(request_id, f"{reduction}_load_time_chunk"):
                embedding_dicts = await asyncio.gather(
                    *[
                        self.apply_in_executor(
                            lambda p: np.load(p, allow_pickle=True).item(),
                            path_tmpl.format(reduction),
                            request_id,
                            f"{reduction}_load_time",
                        )
                        for path_tmpl in path_tmpls
                    ]
                )  # type: List[Dict[int, np.ndarray]]

            # Step 2: Add to applicable on-disk indexes
            for index_name, index in indexes.items():
                with self.profiler(request_id, f"{index_name}_add_time_chunk"):
                    index.add(embedding_dicts)

        return True  # success

    async def process_element(self, *args, **kwargs):
        raise NotImplementedError()

    async def postprocess_chunk(
        self,
        inputs,
        outputs: JSONType,
        job_id,
        job_args,
        request_id,
    ) -> Tuple[str, List[JSONType]]:
        return self.shard_pattern_for_glob, [outputs] * len(inputs)


app = IndexBuildingMapper().server
