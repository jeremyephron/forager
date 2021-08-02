import io
from typing import List, Optional, Tuple

import clip
import numpy as np
import torch
import torchvision.transforms
from detectron2.checkpoint.detection_checkpoint import DetectionCheckpointer
from detectron2.layers import ShapeSpec
from detectron2.modeling.backbone.resnet import build_resnet_backbone
from PIL import Image

from forager_embedding_server.config import CONFIG
from forager_embedding_server.utils import load_remote_file

torch.set_grad_enabled(False)
torch.set_num_threads(1)


class EmbeddingModel:
    def output_dim(self, *args, **kwargs) -> int:
        raise NotImplementedError("Model must implement output_dim")

    def embed_text(self, text: List[str], *args, **kwargs) -> np.ndarray:
        raise NotImplementedError("Model cannot embed text")

    def embed_images(self, images: List[np.ndarray], *args, **kwargs) -> np.ndarray:
        raise NotImplementedError("Model cannot embed images")


class CLIP(EmbeddingModel):
    def __init__(self):
        self.model, self.preprocess = clip.load(
            CONFIG.MODELS.CLIP.MODEL_NAME, device="cpu", jit=True
        )

    def output_dim(self):
        return 512

    def embed_text(self, text, *args, **kwargs):
        text = clip.tokenize(text)
        text_features = self.model.encode_text(text)
        text_features /= text_features.norm(dim=-1, keepdim=True)
        return text_features.numpy()

    def embed_images(self, images, *args, **kwargs):
        preprocessed_images = torch.stack(
            [self.preprocess(Image.fromarray(img)) for img in images]
        )
        image_features = self.model.encode_image(preprocessed_images)
        image_features /= image_features.norm(dim=-1, keepdim=True)
        return image_features.numpy()


class ResNet(EmbeddingModel):
    def __init__(self):
        # Create model
        shape = ShapeSpec(channels=3)
        self.model = torch.nn.Sequential(
            build_resnet_backbone(CONFIG.MODELS.RESNET.CONFIG, shape)
        )

        # Load model weights
        checkpointer = DetectionCheckpointer(self.model, save_to_disk=False)
        checkpointer.load(load_remote_file(CONFIG.MODELS.RESNET.MODEL_URL))
        self.model.eval()

        # Store relevant attributes of config
        self.pixel_mean = torch.tensor(
            CONFIG.MODELS.RESNET.CONFIG.MODEL.PIXEL_MEAN
        ).view(-1, 1, 1)
        self.pixel_std = torch.tensor(CONFIG.MODELS.RESNET.CONFIG.MODEL.PIXEL_STD).view(
            -1, 1, 1
        )
        self.input_format = CONFIG.MODELS.RESNET.CONFIG.INPUT.FORMAT

    def output_dim(self, layer: str = "res4"):
        return {"res4": 1024, "res5": 2048, "linear": 1000}[layer]

    def embed_images(
        self,
        images,
        *args,
        image_patches: Optional[List[Tuple[float, float, float, float]]] = None,
        layer: str = "res4",
        input_size: Tuple[int, int] = (256, 256),
        **kwargs,
    ):
        resize_fn = torchvision.transforms.Resize(input_size, antialias=True)
        converted_images = []
        for idx, img in enumerate(images):
            # Preprocess
            image = Image.fromarray(img).convert("RGB")

            # Crop
            if image_patches and len(image_patches) > idx:
                x1f, y1f, x2f, y2f = image_patches[idx]
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
            image = resize_fn(image)
            converted_images.append(image)

        # Input: NCHW
        # Output: {'res4': CHW, 'res5': CHW} where N = 1
        inputs = torch.stack(converted_images, dim=0)
        output_dict = self.model(inputs)

        return torch.mean(output_dict[layer], dim=[2, 3]).detach().numpy()
