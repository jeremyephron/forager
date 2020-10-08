import io
import os

import aiohttp
from gcloud.aio.storage import Storage
import numpy as np
from PIL import Image
import torch

from detectron2.layers import ShapeSpec
from detectron2.checkpoint.detection_checkpoint import DetectionCheckpointer
from detectron2.modeling.backbone.resnet import build_resnet_backbone

from knn.mappers import Mapper


class ResNetBackboneMapper(Mapper):
    def initialize_container(self, cfg, weights_path):
        # Create model
        shape = ShapeSpec(channels=3)
        self.model = torch.nn.Sequential(build_resnet_backbone(cfg, shape))

        # Load model weights
        checkpointer = DetectionCheckpointer(self.model, save_to_disk=False)
        checkpointer.load(weights_path)
        self.model.eval()
        torch.set_grad_enabled(False)

        # Store relevant attributes of config
        pixel_mean = torch.Tensor(cfg.MODEL.PIXEL_MEAN).view(-1, 1, 1)
        pixel_std = torch.Tensor(cfg.MODEL.PIXEL_STD).view(-1, 1, 1)
        self.normalize = lambda image: (image - pixel_mean) / pixel_std
        self.input_format = cfg.INPUT.FORMAT

        # Create connection pools
        self.session = aiohttp.ClientSession()
        self.storage_client = Storage(session=self.session)

    async def download_and_process_image(self, image_bucket, image_path, request_id):
        # Download image
        async with self.session.get(
            f"https://storage.googleapis.com/{os.path.join(image_bucket, image_path)}"
        ) as response:
            assert response.status == 200
            image_bytes = await response.read()

        # Preprocess image
        with self.profiler(request_id, "compute_time"):
            with io.BytesIO(image_bytes) as image_buffer:
                image = Image.open(image_buffer)

                # Preprocess
                assert image.mode == "RGB"
                image = torch.as_tensor(
                    np.asarray(image), dtype=torch.float32
                )  # -> tensor
                image = image.permute(2, 0, 1)  # HWC -> CHW
                if self.input_format == "BGR":
                    image = torch.flip(image, dims=(0,))  # RGB -> BGR
                image = image.contiguous()
                image = self.normalize(image)

        # Perform inference
        with self.profiler(request_id, "compute_time"):
            result = self.model(image.unsqueeze(dim=0))
            return result
