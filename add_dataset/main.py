import asyncio
import click
import functools
import json
import os
from pathlib import Path
import tempfile
import uuid

import aiohttp
from PIL import Image
from tqdm import tqdm

import clip_inference
import generate_distance_matrix
import resnet_inference


SERVER_URL = os.environ["SERVER_URL"]
CREATE_DATASET_ENDPOINT = "/api/create_dataset_v2"
IMAGE_EXTENSIONS = ("jpg", "jpeg", "png")

INDEX_UPLOAD_GCS_PATH = "gs://foragerml/indexes/"  # trailing slash = directory
THUMBNAIL_UPLOAD_GCS_PATH = "gs://foragerml/thumbnails/"  # trailing slash = directory
RESIZE_THREAD_POOL_SIZE = 10
RESIZE_MAX_HEIGHT = 200


def make_identifier(path):
    return os.path.basename(path).split(".")[0]


def resize_image(input_path, output_dir):
    image = Image.open(input_path)
    image = image.convert("RGB")
    image.thumbnail((image.width, RESIZE_MAX_HEIGHT))
    image.save(output_dir / f"{make_identifier(input_path)}.jpg")


def unasync(coro):
    @functools.wraps(coro)
    def wrapper(*args, **kwargs):
        return asyncio.run(coro(*args, **kwargs))

    return wrapper


@click.command()
@unasync
async def main(name, train_gcs_path, val_gcs_path):
    index_id = str(uuid.uuid4())

    with tempfile.TemporaryDirectory() as parent_dir_name:
        parent_dir = Path(parent_dir_name)

        train_dir = parent_dir / "train"
        val_dir = parent_dir / "val"
        thumbnails_dir = parent_dir / "thumbnails"
        index_dir = parent_dir / "index"
        for d in (train_dir, val_dir, thumbnails_dir, index_dir):
            d.mkdir(parents=True)

        # Download train images
        print("Downloading training images...")
        proc = await asyncio.create_subprocess_exec(
            "gsutil",
            "-m",
            "cp",
            "-r",
            train_gcs_path,
            str(train_dir),
        )
        await proc.wait()
        train_paths = [p for e in IMAGE_EXTENSIONS for p in train_dir.glob(f"*.{e}")]

        # Download val images
        print("Downloading validation images...")
        proc = await asyncio.create_subprocess_exec(
            "gsutil",
            "-m",
            "cp",
            "-r",
            val_gcs_path,
            str(val_dir),
        )
        await proc.wait()
        val_paths = [p for e in IMAGE_EXTENSIONS for p in val_dir.glob(f"*.{e}")]

        # Create identifier files
        train_labels = [str(p.relative_to(train_dir)) for p in train_paths]
        val_labels = [str(p.relative_to(val_dir)) for p in val_paths]

        labels = train_labels + val_labels
        json.dump(labels, Path(index_dir / "labels.json").open("w"))

        train_identifiers = {make_identifier(l): i for i, l in enumerate(train_labels)}
        json.dump(train_identifiers, Path(index_dir / "identifiers.json").open("w"))

        val_identifiers = {
            make_identifier(l): i + len(train_identifiers)
            for i, l in enumerate(val_labels)
        }
        json.dump(val_identifiers, Path(index_dir / "val_identifiers.json").open("w"))

        # Create embeddings
        res4_path = index_dir / "local" / "imagenet_early"
        res5_path = index_dir / "local" / "imagenet"
        clip_path = index_dir / "local" / "clip"
        for d in (res4_path, res5_path, clip_path):
            d.mkdir(parents=True)

        image_paths = train_paths + val_paths
        print("Running ResNet inference...")
        resnet_inference.run(
            image_paths,
            {
                "res4": str(res4_path / "embeddings.npy"),
                "res5": str(res5_path / "embeddings.npy"),
            },
        )

        print("Generating ResNet distance matrix...")
        generate_distance_matrix.run(
            str(res5_path / "embeddings.npy"),
            len(image_paths),
            resnet_inference.EMBEDDING_DIMS["res5"],
            str(res5_path / "distances.npy"),
        )

        print("Running CLIP inference...")
        clip_inference.run(image_paths, str(clip_path / "embeddings.npy"))

        # Upload index to Cloud Storage
        proc = await asyncio.create_subprocess_exec(
            "gsutil",
            "-m",
            "cp",
            "-r",
            str(thumbnails_dir),
            os.path.join(THUMBNAIL_UPLOAD_GCS_PATH, index_id),
        )
        await proc.wait()

        # Create thumbnails
        print("Creating thumbnails...")
        for image_path in tqdm(image_paths):
            resize_image(image_paths, thumbnails_dir)

        # Upload thumbnails to Cloud Storage
        proc = await asyncio.create_subprocess_exec(
            "gsutil",
            "-m",
            "cp",
            "-r",
            str(index_dir),
            os.path.join(INDEX_UPLOAD_GCS_PATH, index_id),
        )
        await proc.wait()

    # Add to database
    params = {
        "dataset": name,
        "train_path": train_gcs_path,
        "val_path": val_gcs_path,
        "index_id": index_id,
    }
    async with aiohttp.ClientSession() as session:
        async with session.post(
            os.path.join(SERVER_URL, CREATE_DATASET_ENDPOINT), json=params
        ) as response:
            j = await response.json()
            assert j["status"] == "success", j


if __name__ == "__main__":
    main()