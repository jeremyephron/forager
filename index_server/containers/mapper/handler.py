import asyncio
import os
from pathlib import Path

import numpy as np

from typing import List, Tuple

from base import ResNetBackboneMapper
from knn.utils import JSONType
import config


class ImageEmbeddingMapper(ResNetBackboneMapper):
    GCS_URL_PREFIX = "https://storage.googleapis.com"

    def initialize_container(self):
        super().initialize_container(config.RESNET_CONFIG, config.WEIGHTS_PATH)

    async def initialize_job(self, job_args):
        job_args["n_chunks_saved"] = 0
        return job_args

    async def process_element(self, *args, **kwargs):
        raise NotImplementedError("Use _process_element instead")

    # Not overriding process_element() because of different return type
    async def _process_element(
        self, input, job_id, job_args, request_id, element_index
    ) -> Tuple[JSONType, np.ndarray]:
        image_path = input["image"]
        if "http" not in image_path:
            image_bucket = job_args["input_bucket"]
            image_path = os.path.join(self.GCS_URL_PREFIX, image_bucket, image_path)

        augmentations = input.get("augmentations", {})
        x1f, y1f, x2f, y2f = input.get("patch", (0, 0, 1, 1))

        spatial_embeddings_dict = await self.download_and_process_image(
            image_path, [x1f, y1f, x2f, y2f], augmentations, request_id
        )

        with self.profiler(request_id, "compute_time"):
            spatial_embeddings = next(iter(spatial_embeddings_dict.values())).numpy()
            n, c, h, w = spatial_embeddings.shape
            return h * w, spatial_embeddings.reshape((c, h * w)).T

    async def process_chunk(
        self, chunk, job_id, job_args, request_id
    ) -> Tuple[str, List[JSONType]]:
        output_embedding_tuples = await asyncio.gather(
            *[
                self._process_element(input, job_id, job_args, request_id, i)
                for i, input in enumerate(chunk)
            ]
        )
        outputs, embeddings = zip(*output_embedding_tuples)

        # Save chunk embeddings dict to disk
        output_path = config.EMBEDDINGS_FILE_PATTERN.format(
            job_id, self.worker_id, job_args["n_chunks_saved"]
        )
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)

        np.save(output_path, dict(zip(chunk, outputs)))
        job_args["n_chunks_saved"] += 1

        return output_path, outputs


mapper = ImageEmbeddingMapper()
