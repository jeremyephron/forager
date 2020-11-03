import asyncio
import io
import os

import aiohttp
from gcloud.aio.storage import Storage
import numpy as np
from PIL import Image, ImageEnhance
import torch
from torchvision.transforms import transforms

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

        # Set up possible augmentations

        def brightness(image, factor):
            br_enhancer = ImageEnhance.Brightness(image)
            return br_enhancer.enhance(factor)
        
        def contrast(image, factor):
            cn_enhancer = ImageEnhance.Contrast(image)
            return cn_enhancer.enhance(factor)

        self.flip = lambda image: image.transpose(Image.FLIP_LEFT_RIGHT)
        self.grayscale = lambda image: image.convert('L')
        self.brightness = brightness
        self.contrast = contrast
        self.resize = lambda image, params: image.resize(((int)(params*image.size[0]), (int)(params*image.size[1])))
        self.rotate = lambda image, angle: image.rotate(angle)

        # Create connection pools
        self.session = aiohttp.ClientSession()
        self.storage_client = Storage(session=self.session)

    async def download_and_process_image(
        self, image_bucket, image_path, image_patch, augmentations, request_id, num_retries=4
    ):
        # Download image
        for i in range(num_retries):
            try:
                async with self.session.get(
                    f"https://storage.googleapis.com/{os.path.join(image_bucket, image_path)}"
                ) as response:
                    assert response.status == 200
                    image_bytes = await response.read()
            except Exception:
                if i < num_retries - 1:
                    await asyncio.sleep(2 ** i)
                else:
                    raise

        # Preprocess image
        with self.profiler(request_id, "compute_time"):
            with io.BytesIO(image_bytes) as image_buffer:
                image = Image.open(image_buffer)

                # Preprocess
                image = image.convert("RGB")

                # Crop
                if image_patch:
                    print("Image patch:")
                    print(image_patch)
                    x1f, y1f, x2f, y2f = image_patch
                    w, h = image.size
                    image = image.crop(((int)(x1f*w), (int)(y1f*h), (int)(x2f*w), (int)(y2f*h)))

                # Apply transformations (augmentations is a dict)
                if ("flip" in augmentations):  
                    image = self.flip(image)
                if ("gray" in augmentations):
                    image = self.grayscale(image)
                if ("brightness" in augmentations):
                    image = self.brightness(image, augmentations["brightness"])
                if ("contrast" in augmentations):
                    image = self.contrast(image, augmentations["contrast"])
                if ("resize" in augmentations):
                    image = self.rescale(image, augmentations["resize"])
                if ("rotate" in augmentations):
                    image = self.rotate(image, augmentations["rotate"])

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
            # Input: NCHW
            # Output: {'res4': NCHW, 'res5': NCHW} where N = 1
            return self.model(image.unsqueeze(dim=0))
