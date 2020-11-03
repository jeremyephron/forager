import math
import torch

from knn import utils

from knn.mappers import Mapper
from base import ResNetBackboneMapper
import config


class ImageEmbeddingMapper(ResNetBackboneMapper):
    # input, job args govern data
    async def process_element(self, input, job_id, job_args, request_id, element_index):
        image_bucket = job_args["input_bucket"]
        image_path = input["image"]
        augmentations = input["augmentations"]
        x1f, y1f, x2f, y2f = input.get("patch", (0, 0, 1, 1))
        print(input)

        spatial_embeddings_dict = await self.download_and_process_image(
            image_bucket, image_path, [x1f, y1f, x2f, y2f], augmentations, request_id
        )

        with self.profiler(request_id, "compute_time"):
            result = {}

            for layer, spatial_embeddings in spatial_embeddings_dict.items():
                n, c, h, w = spatial_embeddings.size()
                assert n == 1

                # Come back and revisit whether to do this
                # x1 = int(math.floor(x1f * w))
                # y1 = int(math.floor(y1f * h))
                # x2 = int(math.ceil(x2f * w))
                # y2 = int(math.ceil(y2f * h))

                cropped = spatial_embeddings[0, :, y1:y2, x1:x2]

                # temporary fix for OOM with spatial embeddings
                cropped = torch.mean(cropped, dim=(1, 2), keepdim=True)

                result[layer] = utils.numpy_to_base64(cropped.numpy())

            return result


mapper = ImageEmbeddingMapper(config.RESNET_CONFIG, config.WEIGHTS_PATH)
