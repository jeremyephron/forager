import asyncio
import concurrent
from enum import Enum
import os
from pathlib import Path

import aiohttp
import numpy as np
import torch

from detectron2.layers import ShapeSpec
from detectron2.checkpoint.detection_checkpoint import DetectionCheckpointer
from detectron2.modeling.backbone.resnet import build_resnet_backbone

from typing import List, Optional, Tuple, Union

from knn import utils
from knn.mappers import Mapper

import config
import inference


class IndexEmbeddingMapper(Mapper):
    class ReturnType(Enum):
        SAVE = 0
        SERIALIZE = 1

    def initialize_container(self):
        torch.set_grad_enabled(False)
        torch.set_num_threads(1)

        # Create model
        shape = ShapeSpec(channels=3)
        self.model = torch.nn.Sequential(
            build_resnet_backbone(config.RESNET_CONFIG, shape)
        )

        # Load model weights
        checkpointer = DetectionCheckpointer(self.model, save_to_disk=False)
        checkpointer.load(config.WEIGHTS_PATH)
        self.model.eval()

        # Store relevant attributes of config
        self.pixel_mean = torch.Tensor(config.RESNET_CONFIG.MODEL.PIXEL_MEAN).view(
            -1, 1, 1
        )
        self.pixel_std = torch.Tensor(config.RESNET_CONFIG.MODEL.PIXEL_STD).view(
            -1, 1, 1
        )
        self.input_format = config.RESNET_CONFIG.INPUT.FORMAT

        # Create connection pool
        self.session = aiohttp.ClientSession()

        # Create inference pool
        if config.NUM_CPUS > 1:
            self.model.share_memory()
            self.pixel_mean.share_memory_()
            self.pixel_std.share_memory_()
            self.pool_executor = concurrent.futures.ProcessPoolExecutor(
                config.NUM_CPUS, mp_context=torch.multiprocessing.get_context("spawn")
            )
        else:
            self.pool_executor = None

    async def initialize_job(self, job_args):
        return_type = job_args.get("return_type", "serialize")
        if return_type == "save":
            job_args["return_type"] = IndexEmbeddingMapper.ReturnType.SAVE
        elif return_type == "serialize":
            job_args["return_type"] = IndexEmbeddingMapper.ReturnType.SERIALIZE
        else:
            raise ValueError(f"Unknown return type: {return_type}")

        job_args["n_chunks_saved"] = 0
        return job_args

    @utils.log_exception_from_coro_but_return_none
    async def process_element(
        self, input, job_id, job_args, request_id, element_index
    ) -> np.ndarray:
        image_path = input["image"]
        image_patch = input.get("patch", (0, 0, 1, 1))
        augmentations = input.get("augmentations", {})

        # Download image
        if "http" not in image_path:
            image_bucket = job_args["input_bucket"]
            image_path = os.path.join(config.GCS_URL_PREFIX, image_bucket, image_path)
        with self.profiler(request_id, "download_time"):
            image_bytes = await self.download_image(image_path)

        # Run inference
        inference_args = (
            image_bytes,
            image_patch,
            augmentations,
            self.input_format,
            self.pixel_mean,
            self.pixel_std,
            self.model,
        )
        with self.profiler(request_id, "inference_time"):
            if self.pool_executor:
                loop = asyncio.get_running_loop()
                model_output_dict = await loop.run_in_executor(
                    self.pool_executor,
                    inference.run,
                    *inference_args,
                )
            else:
                model_output_dict = inference.run(*inference_args)

        with self.profiler(request_id, "flatten_time"):
            spatial_embeddings = next(iter(model_output_dict.values())).numpy()
            n, c, h, w = spatial_embeddings.shape
            assert n == 1
            return spatial_embeddings.reshape((c, h * w)).T

    async def postprocess_chunk(
        self,
        inputs,
        outputs: List[Optional[np.ndarray]],
        job_id,
        job_args,
        request_id,
    ) -> Union[Tuple[str, List[Optional[int]]], Tuple[None, List[Optional[str]]]]:
        if job_args["return_type"] == IndexEmbeddingMapper.ReturnType.SAVE:
            with self.profiler(request_id, "save_time"):
                # Save chunk embeddings dict to disk
                output_path = config.EMBEDDINGS_FILE_TMPL.format(
                    job_id, self.worker_id, job_args["n_chunks_saved"]
                )
                embeddings_dict = {
                    int(input["id"]): output
                    for input, output in zip(inputs, outputs)
                    if output is not None
                }

                Path(output_path).parent.mkdir(parents=True, exist_ok=True)
                np.save(output_path, embeddings_dict)
                job_args["n_chunks_saved"] += 1

                return output_path, [
                    len(output) if output is not None else None for output in outputs
                ]
        else:
            return None, [
                utils.numpy_to_base64(output) if output is not None else None
                for output in outputs
            ]


mapper = IndexEmbeddingMapper()
