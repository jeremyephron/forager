from knn import utils

from knn.mappers import Mapper
from base import ResNetBackboneMapper
import config


class ImageEmbeddingMapper(ResNetBackboneMapper):
    @Mapper.SkipIfError
    async def process_element(self, input, job_id, job_args, request_id, element_index):
        image_bucket = job_args["input_bucket"]
        image_path = input

        spatial_embeddings = await self.download_and_process_image(
            image_bucket, image_path, request_id
        )

        with self.profiler(request_id, "compute_time"):
            return {
                k: utils.numpy_to_base64(v.mean(dim=-1).mean(dim=-1).numpy())
                for k, v in spatial_embeddings.items()
            }


mapper = ImageEmbeddingMapper(config.RESNET_CONFIG, config.WEIGHTS_PATH)
