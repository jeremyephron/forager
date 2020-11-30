import os
from pathlib import Path

import numpy as np

from typing import List, Tuple

from base import ResNetBackboneMapper
from knn.utils import JSONType
import config


class IndexEmbeddingMapper(ResNetBackboneMapper):
    def initialize_container(self):
        super().initialize_container(config.RESNET_CONFIG, config.WEIGHTS_PATH)

    async def initialize_job(self, job_args):
        job_args["n_chunks_saved"] = 0
        return job_args

    async def process_element(
        self, input, job_id, job_args, request_id, element_index
    ) -> np.ndarray:
        image_path = input["image"]
        if "http" not in image_path:
            image_bucket = job_args["input_bucket"]
            image_path = os.path.join(config.GCS_URL_PREFIX, image_bucket, image_path)

        augmentations = input.get("augmentations", {})
        x1f, y1f, x2f, y2f = input.get("patch", (0, 0, 1, 1))

        model_output_dict = await self.download_and_process_image(
            image_path, [x1f, y1f, x2f, y2f], augmentations, request_id
        )

        with self.profiler(request_id, "compute_time"):
            spatial_embeddings = next(iter(model_output_dict.values())).numpy()
            n, c, h, w = spatial_embeddings.shape
            assert n == 1
            return spatial_embeddings.reshape((c, h * w)).T

    async def postprocess_chunk(
        self,
        inputs,
        outputs: List[np.ndarray],
        job_id,
        job_args,
        request_id,
    ) -> Tuple[str, List[int]]:
        # Save chunk embeddings dict to disk
        output_path = config.EMBEDDINGS_FILE_TMPL.format(
            job_id, self.worker_id, job_args["n_chunks_saved"]
        )
        embeddings_dict = {
            int(input["id"]): embeddings for input, embeddings in zip(inputs, outputs)
        }

        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        np.save(output_path, embeddings_dict)
        job_args["n_chunks_saved"] += 1

        return output_path, list(map(len, outputs))


mapper = IndexEmbeddingMapper()
