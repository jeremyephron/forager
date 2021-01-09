import asyncio
from collections import ChainMap, defaultdict

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

    def add(self, embedding_dict: Dict[int, np.ndarray]) -> int:
        ids = [
            int(id)
            for id, embeddings in embedding_dict.items()
            for _ in range(embeddings.shape[0])
        ]
        self.index.add(
            np.concatenate(list(embedding_dict.values())), ids, update_metadata=False
        )
        return len(ids)


class IndexBuildingMapper(Mapper):
    def initialize_container(self):
        self.shard_pattern_for_glob = config.SHARD_INDEX_NAME_TMPL.format(
            self.worker_id
        ).format("*")

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

    # input = path to a np.save'd Dict[int, np.ndarray] where each value is N x D
    @utils.log_exception_from_coro_but_return_none
    async def process_element(
        self, input, job_id, job_args, request_id, element_index
    ) -> Dict[str, int]:
        indexes_by_reduction = job_args["indexes_by_reduction"]
        path_tmpl = input

        num_added_dicts = await asyncio.gather(
            *[
                self.build_indexes_for_reduction(
                    reduction, path_tmpl, indexes, request_id
                )
                for reduction, indexes in indexes_by_reduction.items()
            ]
        )
        return dict(ChainMap(*num_added_dicts))  # merge the dicts

    async def build_indexes_for_reduction(
        self,
        reduction: Optional[str],
        path_tmpl: str,
        indexes: Dict[str, Index],
        request_id: str,
    ) -> Dict[str, int]:
        # Step 1: Load saved embeddings into memory
        with self.profiler(request_id, f"{reduction}_load_time"):
            embedding_dict = np.load(
                path_tmpl.format(reduction), allow_pickle=True
            ).item()  # type: Dict[int, np.ndarray]

        # Step 2: Add to applicable on-disk indexes
        num_added = {}
        for index_name, index in indexes.items():
            with self.profiler(request_id, f"{index_name}_add_time"):
                num_added[index_name] = index.add(embedding_dict)
        return num_added

    async def postprocess_chunk(
        self,
        inputs,
        outputs,
        job_id,
        job_args,
        request_id,
    ) -> Tuple[str, List[JSONType]]:
        return self.shard_pattern_for_glob, outputs


app = IndexBuildingMapper().server
