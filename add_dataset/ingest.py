import asyncio
import json
import os
from pathlib import Path
import uuid
import numpy as np
import time
import pickle

import aiohttp
import click
from PIL import Image
from tqdm import tqdm
from datetime import timedelta

import clip_inference
import resnet_inference
from utils import make_identifier, parse_gcs_path, unasync


SERVER_URL = os.environ["SERVER_URL"]
GET_DATASET_ENDPOINT = "api/get_dataset_info_v2"
CREATE_DATASET_ENDPOINT = "api/create_dataset_v2"
IMAGE_EXTENSIONS = ("jpg", "jpeg", "png")

INDEX_UPLOAD_GCS_PATH = "gs://foragerml/indexes/"  # trailing slash = directory
AUX_LABELS_UPLOAD_GCS_PATH = "gs://foragerml/aux_labels/"  # trailing slash = directory
THUMBNAIL_UPLOAD_GCS_PATH = "gs://foragerml/thumbnails/"  # trailing slash = directory
RESIZE_MAX_HEIGHT = 200


def resize_image(input_path, output_dir):
    image = Image.open(input_path)
    image = image.convert("RGB")
    image.thumbnail((image.width, RESIZE_MAX_HEIGHT))
    image.save(output_dir / f"{make_identifier(input_path)}.jpg")


@click.command()
@click.argument("name")
@click.argument("train_gcs_path")
@click.argument("val_gcs_path")
@click.option("--resnet_full_batch_size", type=int, default=1)
@click.option("--resnet_batch_size", type=int, default=16)
@click.option("--resnet_resize_size", type=int, default=256)
@click.option("--resnet_crop_size", type=int, default=224)
@click.option("--use_proxy", is_flag=True)
@unasync
async def main(
    name,
    train_gcs_path,
    val_gcs_path,
    resnet_full_batch_size,
    resnet_batch_size,
    resnet_resize_size,
    resnet_crop_size,
    use_proxy
):
    if not use_proxy and ('http_proxy' in os.environ or
                          'https_proxy' in os.environ):
        print('WARNING: http_proxy/https_proxy env variables set, but '
              '--use_proxy flag not specified. Will not use proxy.')

    # Make sure that a dataset with this name doesn't already exist
    async with aiohttp.ClientSession(trust_env=use_proxy) as session:
        async with session.get(
            os.path.join(SERVER_URL, GET_DATASET_ENDPOINT, name)
        ) as response:
            assert response.status == 404, f"Dataset {name} already exists"

    index_id = str(uuid.uuid4())

    parent_dir = Path() / name

    train_dir = parent_dir / "train"
    val_dir = parent_dir / "val"
    thumbnails_dir = parent_dir / "thumbnails"
    index_dir = parent_dir / "index"
    aux_labels_dir = parent_dir / "aux_labels"
    for d in (train_dir, val_dir, thumbnails_dir, index_dir, aux_labels_dir):
        d.mkdir(parents=True, exist_ok=True)

    # Download train images
    download_start = time.time()
    if True:
        print("Downloading training images...")
        proc = await asyncio.create_subprocess_exec(
            "gsutil",
            "-m",
            "cp",
            "-r",
            "-n",
            os.path.join(train_gcs_path, "*"),
            str(train_dir),
        )
        await proc.wait()
    train_paths = [p for e in IMAGE_EXTENSIONS for p in train_dir.glob(f"**/*.{e}")]
    # Download val images
    if True:
        print("Downloading validation images...")
        proc = await asyncio.create_subprocess_exec(
            "gsutil",
            "-m",
            "cp",
            "-r",
            "-n",
            os.path.join(val_gcs_path, "*"),
            str(val_dir),
        )
        await proc.wait()
    download_end = time.time()

    val_paths = [p for e in IMAGE_EXTENSIONS for p in val_dir.glob(f"**/*.{e}")]

    # Create identifier files
    _, train_gcs_relative_path = parse_gcs_path(train_gcs_path)
    _, val_gcs_relative_path = parse_gcs_path(val_gcs_path)

    train_labels = [
        os.path.join(train_gcs_relative_path, p.relative_to(train_dir))
        for p in train_paths
    ]
    val_labels = [
        os.path.join(val_gcs_relative_path, p.relative_to(val_dir)) for p in val_paths
    ]

    labels = train_labels + val_labels
    json.dump(labels, Path(index_dir / "labels.json").open("w"))

    train_identifiers = {make_identifier(l): i for i, l in enumerate(train_labels)}
    json.dump(train_identifiers, Path(index_dir / "identifiers.json").open("w"))

    val_identifiers = {
        make_identifier(l): i + len(train_identifiers) for i, l in enumerate(val_labels)
    }
    json.dump(val_identifiers, Path(index_dir / "val_identifiers.json").open("w"))

    # Create embeddings
    res4_path = index_dir / "local" / "imagenet_early"
    res5_path = index_dir / "local" / "imagenet"
    linear_path = index_dir / "local" / "imagenet_linear"

    res4_full_path = index_dir / "local" / "imagenet_full_early"
    res5_full_path = index_dir / "local" / "imagenet_full"

    clip_path = index_dir / "local" / "clip"
    for d in (
        res4_path,
        res5_path,
        clip_path,
        linear_path,
        res4_full_path,
        res5_full_path,
    ):
        d.mkdir(parents=True, exist_ok=True)

    image_paths = train_paths + val_paths

    resnet_layers = {
        "res4": str(res4_path / "embeddings.npy"),
        "res5": str(res5_path / "embeddings.npy"),
        "linear": str(linear_path / "embeddings.npy"),
    }
    resnet_full_layers = {
        "res4": str(res4_full_path / "embeddings.npy"),
        "res5": str(res5_full_path / "embeddings.npy"),
    }
    resnet_start = time.time()
    if True:
        print("Running ResNet inference...")
        # Run at cropped resnet size
        resnet_inference.run(
            image_paths,
            resnet_layers,
            batch_size=resnet_batch_size,
            resize_to=resnet_resize_size,
            crop_to=resnet_crop_size,
        )
        # Copy aux labels
        linear_embeddings = np.memmap(
            resnet_layers["linear"],
            dtype="float32",
            mode="r",
            shape=(len(image_paths), resnet_inference.EMBEDDING_DIMS["linear"]),
        )
        model_labels = np.argmax(linear_embeddings, axis=1)
        aux_labels = {}
        for label, aux_l in zip(labels, model_labels[:]):
            aux_labels[os.path.basename(label)] = aux_l
        with open(str(aux_labels_dir / "imagenet.pickle"), "wb") as f:
            pickle.dump(aux_labels, f)

    if True:
        # Run at full resolution
        resnet_inference.run(
            image_paths,
            resnet_full_layers,
            batch_size=resnet_full_batch_size,
        )

    resnet_end = time.time()

    clip_start = time.time()
    if True:
        print("Running CLIP inference...")
        clip_inference.run(image_paths, str(clip_path / "embeddings.npy"))
    clip_end = time.time()

    # Create thumbnails
    thumbnail_start = time.time()
    if True:
        print("Creating thumbnails...")
        for path in tqdm(image_paths):
            resize_image(path, thumbnails_dir)
    thumbnail_end = time.time()

    # Upload index to Cloud Storage
    upload_start = time.time()
    if True:
        proc = await asyncio.create_subprocess_exec(
            "gsutil",
            "-m",
            "cp",
            "-r",
            str(index_dir),
            os.path.join(INDEX_UPLOAD_GCS_PATH, index_id),
        )
        await proc.wait()

        proc = await asyncio.create_subprocess_exec(
            "gsutil",
            "-m",
            "cp",
            "-r",
            str(aux_labels_dir),
            os.path.join(AUX_LABELS_UPLOAD_GCS_PATH, index_id),
        )
        await proc.wait()

        # Upload thumbnails to Cloud Storage
        proc = await asyncio.create_subprocess_exec(
            "gsutil",
            "-m",
            "cp",
            "-r",
            str(thumbnails_dir),
            os.path.join(THUMBNAIL_UPLOAD_GCS_PATH, index_id),
        )
        await proc.wait()
    upload_end = time.time()

    # Add to database
    params = {
        "dataset": name,
        "train_path": train_gcs_path,
        "val_path": val_gcs_path,
        "index_id": index_id,
    }
    add_db_start = time.time()
    if True:
        async with aiohttp.ClientSession(trust_env=use_proxy) as session:
            async with session.post(
                os.path.join(SERVER_URL, CREATE_DATASET_ENDPOINT), json=params
            ) as response:
                j = await response.json()
                assert j["status"] == "success", j
    add_db_end = time.time()

    print("Timing")
    for k, t in [
        ("Download", download_end - download_start),
        ("Resnet", resnet_end - resnet_start),
        ("Clip", clip_end - clip_start),
        ("Thumbnail", thumbnail_end - thumbnail_start),
        ("Upload", upload_end - upload_start),
        ("Add db", add_db_end - add_db_start),
    ]:
        print("{:15} {}".format(k, str(timedelta(seconds=t))))


if __name__ == "__main__":
    main()
