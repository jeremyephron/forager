import io

from typing import List, Optional

import clip
from detectron2.layers import ShapeSpec
from detectron2.checkpoint.detection_checkpoint import DetectionCheckpointer
from detectron2.modeling.backbone.resnet import build_resnet_backbone
import numpy as np
from PIL import Image
import torch

import config
from utils import load_remote_file


torch.set_grad_enabled(False)
torch.set_num_threads(1)


class EmbeddingModel:
    def embed_text(self, text: str, *args, **kwargs) -> np.ndarray:
        raise NotImplementedError("Model cannot embed text")

    def embed_image_bytes(self, image_bytes: bytes, *args, **kwargs) -> np.ndarray:
        raise NotImplementedError("Model cannot embed images")


class CLIP(EmbeddingModel):
    def __init__(self):
        self.model = clip.load(load_remote_file(config.CLIP_MODEL_URL), device="cpu")

    def embed_text(self, text, *args, **kwargs):
        text = clip.tokenize([text])
        text_features = self.model.encode_text(text).squeeze(0)
        text_features /= text_features.norm(dim=-1, keepdim=True)
        return text_features.numpy()


class ResNet(EmbeddingModel):
    def __init__(self):
        # Create model
        shape = ShapeSpec(channels=3)
        self.model = torch.nn.Sequential(
            build_resnet_backbone(config.RESNET_CONFIG, shape)
        )

        # Load model weights
        checkpointer = DetectionCheckpointer(self.model, save_to_disk=False)
        checkpointer.load(load_remote_file(config.RESNET_MODEL_URL))
        self.model.eval()

        # Store relevant attributes of config
        self.pixel_mean = torch.tensor(config.RESNET_CONFIG.MODEL.PIXEL_MEAN).view(
            -1, 1, 1
        )
        self.pixel_std = torch.tensor(config.RESNET_CONFIG.MODEL.PIXEL_STD).view(
            -1, 1, 1
        )
        self.input_format = config.RESNET_CONFIG.INPUT.FORMAT

    def embed_image_bytes(
        self,
        image_bytes,
        *args,
        image_patch: Optional[List[float]] = None,
        layer: str = "res4",
        **kwargs,
    ):
        with io.BytesIO(image_bytes) as image_buffer:
            image = Image.open(image_buffer)

            # Preprocess
            image = image.convert("RGB")

            # Crop
            if image_patch:
                x1f, y1f, x2f, y2f = image_patch
                w, h = image.size
                image = image.crop(
                    ((int)(x1f * w), (int)(y1f * h), (int)(x2f * w), (int)(y2f * h))
                )

            image = torch.as_tensor(np.asarray(image), dtype=torch.float32)  # -> tensor
            image = image.permute(2, 0, 1)  # HWC -> CHW
            if self.input_format == "BGR":
                image = torch.flip(image, dims=(0,))  # RGB -> BGR
            image = image.contiguous()
            image = (image - self.pixel_mean) / self.pixel_std

        # Input: NCHW
        # Output: {'res4': CHW, 'res5': CHW} where N = 1
        output_dict = self.model(image.unsqueeze(dim=0))

        return output_dict[layer].detach().squeeze(0).numpy()
