import functools

import numpy as np

from interactive_index import InteractiveIndex
from knn.mappers import Mapper
import config


class IndexBuildingMapper(Mapper):
    async def initialize_job(self, job_args) -> InteractiveIndex:
        reduction = job_args.get("reduction")
        if reduction == "average":
            job_args["reduction"] = functools.partial(np.mean, axis=1, keepdims=True)
        elif not reduction:
            job_args["reduction"] = None
        else:
            raise ValueError(f"Unknown reduction: {reduction}")

        index = InteractiveIndex.load(job_args["index_dir"])
        index.SHARD_INDEX_NAME_TMPL = config.SHARD_INDEX_NAME_TMPL.format(
            self.worker_id
        )
        job_args["index"] = index

        return job_args

    # input = path to a np.save'd Dict[int, np.ndarray] where each value is N x D
    async def process_element(
        self, input, job_id, job_args, request_id, element_index
    ) -> int:
        reduction = job_args["reduction"] or (lambda x: x)
        index = job_args["index"]

        # Step 1: Load saved embeddings into memory
        embedding_dict = np.load(
            input, allow_pickle=True
        ).item()  # type: Dict[int, np.ndarray]

        # Step 2: Add to on-disk index
        all_embeddings = list(map(reduction, embedding_dict.values()))
        all_ids = [
            int(id)
            for id, embeddings in zip(embedding_dict, all_embeddings)
            for _ in range(embeddings.shape[0])
        ]
        index.add(np.concatenate(all_embeddings), all_ids)

        return len(all_ids)

    async def postprocess_chunk(
        self,
        inputs,
        outputs,
        job_id,
        job_args,
        request_id,
    ) -> Tuple[str, List[JSONType]]:
        return job_args["index"].SHARD_INDEX_NAME_TMPL, outputs


mapper = IndexBuildingMapper()
