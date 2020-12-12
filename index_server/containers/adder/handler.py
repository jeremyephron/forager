import functools

import numpy as np

from typing import Dict, List, Tuple

from interactive_index import InteractiveIndex
from knn.mappers import Mapper
from knn.utils import JSONType
import config


class Index:
    def __init__(self, index_dict: Dict[str, str], shard_tmpl: str):
        reduction = index_dict.get("reduction")
        index_dir = index_dict["index_dir"]

        if reduction == "average":
            self.reduction = functools.partial(np.mean, axis=1, keepdims=True)
        elif not reduction:
            self.reduction = lambda x: x
        else:
            raise ValueError(f"Unknown reduction: {reduction}")

        self.index = InteractiveIndex.load(index_dir)
        self.index.SHARD_INDEX_NAME_TMPL = shard_tmpl

    def add(self, embedding_dict: Dict[int, np.ndarray]):
        all_embeddings = list(map(self.reduction, embedding_dict.values()))
        all_ids = [
            int(id)
            for id, embeddings in zip(embedding_dict, all_embeddings)
            for _ in range(embeddings.shape[0])
        ]
        self.index.add(np.concatenate(all_embeddings), all_ids)

        return len(all_ids)


class IndexBuildingMapper(Mapper):
    def initialize_container(self):
        self.shard_tmpl = config.SHARD_INDEX_NAME_TMPL.format(self.worker_id)

    async def initialize_job(self, job_args) -> InteractiveIndex:
        job_args["indexes"] = {
            index_type: Index(index_dict, self.shard_tmpl)
            for index_type, index_dict in job_args["indexes"].items()
        }
        return job_args

    # input = path to a np.save'd Dict[int, np.ndarray] where each value is N x D
    async def process_element(
        self, input, job_id, job_args, request_id, element_index
    ) -> List[int]:
        # Step 1: Load saved embeddings into memory
        embedding_dict = np.load(
            input, allow_pickle=True
        ).item()  # type: Dict[int, np.ndarray]

        # Step 2: Add to on-disk indexes
        return {
            index_type: index.add(embedding_dict)
            for index_type, index in job_args["indexes"].items()
        }

    async def postprocess_chunk(
        self,
        inputs,
        outputs,
        job_id,
        job_args,
        request_id,
    ) -> Tuple[str, List[JSONType]]:
        return self.shard_tmpl, outputs


mapper = IndexBuildingMapper()
