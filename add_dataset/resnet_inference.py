import concurrent.futures

from typing import List, Dict, Optional

import numpy as np
import torch
import torchvision.transforms.functional as tvf
import functools
from PIL import Image
from tqdm import tqdm

from detectron2.layers import ShapeSpec
from detectron2.checkpoint.detection_checkpoint import DetectionCheckpointer
from detectron2.modeling.backbone.resnet import (
    ResNet,
    BasicStem,
    DeformBottleneckBlock,
    BottleneckBlock,
    BasicBlock,
)
from detectron2.config.config import get_cfg as get_default_detectron_config


def build_resnet_backbone(cfg, input_shape, num_classes=None):
    """
    Create a ResNet instance from config.
    Returns:
        ResNet: a :class:`ResNet` instance.
    """
    # need registration of new blocks/stems?
    norm = cfg.MODEL.RESNETS.NORM
    stem = BasicStem(
        in_channels=input_shape.channels,
        out_channels=cfg.MODEL.RESNETS.STEM_OUT_CHANNELS,
        norm=norm,
    )

    # fmt: off
    freeze_at           = cfg.MODEL.BACKBONE.FREEZE_AT
    out_features        = cfg.MODEL.RESNETS.OUT_FEATURES
    depth               = cfg.MODEL.RESNETS.DEPTH
    num_groups          = cfg.MODEL.RESNETS.NUM_GROUPS
    width_per_group     = cfg.MODEL.RESNETS.WIDTH_PER_GROUP
    bottleneck_channels = num_groups * width_per_group
    in_channels         = cfg.MODEL.RESNETS.STEM_OUT_CHANNELS
    out_channels        = cfg.MODEL.RESNETS.RES2_OUT_CHANNELS
    stride_in_1x1       = cfg.MODEL.RESNETS.STRIDE_IN_1X1
    res5_dilation       = cfg.MODEL.RESNETS.RES5_DILATION
    deform_on_per_stage = cfg.MODEL.RESNETS.DEFORM_ON_PER_STAGE
    deform_modulated    = cfg.MODEL.RESNETS.DEFORM_MODULATED
    deform_num_groups   = cfg.MODEL.RESNETS.DEFORM_NUM_GROUPS
    # fmt: on
    assert res5_dilation in {1, 2}, "res5_dilation cannot be {}.".format(res5_dilation)

    num_blocks_per_stage = {
        18: [2, 2, 2, 2],
        34: [3, 4, 6, 3],
        50: [3, 4, 6, 3],
        101: [3, 4, 23, 3],
        152: [3, 8, 36, 3],
    }[depth]

    if depth in [18, 34]:
        assert (
            out_channels == 64
        ), "Must set MODEL.RESNETS.RES2_OUT_CHANNELS = 64 for R18/R34"
        assert not any(
            deform_on_per_stage
        ), "MODEL.RESNETS.DEFORM_ON_PER_STAGE unsupported for R18/R34"
        assert (
            res5_dilation == 1
        ), "Must set MODEL.RESNETS.RES5_DILATION = 1 for R18/R34"
        assert num_groups == 1, "Must set MODEL.RESNETS.NUM_GROUPS = 1 for R18/R34"

    stages = []

    for idx, stage_idx in enumerate(range(2, 6)):
        # res5_dilation is used this way as a convention in R-FCN & Deformable Conv paper
        dilation = res5_dilation if stage_idx == 5 else 1
        first_stride = 1 if idx == 0 or (stage_idx == 5 and dilation == 2) else 2
        stage_kargs = {
            "num_blocks": num_blocks_per_stage[idx],
            "stride_per_block": [first_stride] + [1] * (num_blocks_per_stage[idx] - 1),
            "in_channels": in_channels,
            "out_channels": out_channels,
            "norm": norm,
        }
        # Use BasicBlock for R18 and R34.
        if depth in [18, 34]:
            stage_kargs["block_class"] = BasicBlock
        else:
            stage_kargs["bottleneck_channels"] = bottleneck_channels
            stage_kargs["stride_in_1x1"] = stride_in_1x1
            stage_kargs["dilation"] = dilation
            stage_kargs["num_groups"] = num_groups
            if deform_on_per_stage[idx]:
                stage_kargs["block_class"] = DeformBottleneckBlock
                stage_kargs["deform_modulated"] = deform_modulated
                stage_kargs["deform_num_groups"] = deform_num_groups
            else:
                stage_kargs["block_class"] = BottleneckBlock
        blocks = ResNet.make_stage(**stage_kargs)
        in_channels = out_channels
        out_channels *= 2
        bottleneck_channels *= 2
        stages.append(blocks)
    return ResNet(
        stem,
        stages,
        num_classes=num_classes,
        out_features=out_features,
        freeze_at=freeze_at,
    )


EMBEDDING_DIMS = {"res4": 1024, "res5": 2048, "linear": 1000}
WEIGHTS_PATH = "R-50.pkl"
RESNET_CONFIG = get_default_detectron_config()
RESNET_CONFIG.MODEL.RESNETS.OUT_FEATURES = list(EMBEDDING_DIMS.keys())


# Create model
shape = ShapeSpec(channels=3)
backbone = build_resnet_backbone(
    RESNET_CONFIG, shape, num_classes=EMBEDDING_DIMS["linear"]
)
model = torch.nn.Sequential(backbone)


# Load model
device = "cuda" if torch.cuda.is_available() else "cpu"
checkpointer = DetectionCheckpointer(model, save_to_disk=False)
checkpointer.load(WEIGHTS_PATH)
model.to(device)
model.eval()

# model = torch.nn.DataParallel(model)

pixel_mean = torch.tensor(RESNET_CONFIG.MODEL.PIXEL_MEAN).view(-1, 1, 1)
pixel_std = torch.tensor(RESNET_CONFIG.MODEL.PIXEL_STD).view(-1, 1, 1)
input_format = RESNET_CONFIG.INPUT.FORMAT


def load_image(resize_size, crop_size, path):
    image = Image.open(path)
    image = image.convert("RGB")
    image = torch.as_tensor(np.asarray(image), dtype=torch.float32)  # -> tensor
    image = image.permute(2, 0, 1)  # HWC -> CHW
    if input_format == "BGR":
        image = torch.flip(image, dims=(0,))  # RGB -> BGR
    image = image.contiguous()
    image = (image - pixel_mean) / pixel_std
    image = image.unsqueeze(dim=0)
    if resize_size:
        image = tvf.resize(image, resize_size)
    if crop_size:
        image = tvf.center_crop(image, crop_size)
    return image.to(device)


def run(
    image_paths: List[str],
    embeddings_output_filenames: Dict[str, str],
    batch_size: int = 1,
    resize_to: Optional[List[int]] = None,
    crop_to: Optional[List[int]] = None,
):
    embeddings = {
        layer: np.memmap(
            embeddings_output_filenames[layer],
            dtype="float32",
            mode="w+",
            shape=(len(image_paths), dim),
        )
        for layer, dim in EMBEDDING_DIMS.items()
    }

    with concurrent.futures.ThreadPoolExecutor() as executor:
        for i in tqdm(range(0, len(image_paths), batch_size)):
            # Load batch of images
            batch_paths = image_paths[i : i + batch_size]
            images = torch.cat(
                list(
                    executor.map(
                        functools.partial(load_image, resize_to, crop_to), batch_paths
                    )
                )
            )

            with torch.no_grad():
                output_dict = model(images)
                for layer, e in embeddings.items():
                    if layer == "linear":
                        outs = output_dict[layer]
                        e[i : i + batch_size] = outs.cpu().numpy()
                    else:
                        outs = output_dict[layer].mean(dim=(2, 3))
                        outs /= outs.norm(dim=-1, keepdim=True)
                        e[i : i + batch_size] = outs.cpu().numpy()

    for e in embeddings.values():
        e.flush()
