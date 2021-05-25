# Adding a dataset to Forager

This script allows you to ingest data into a running Forager instance. Specifically, it turns directories of training and validation images on Google Cloud Storage into a Forager "dataset" that can be explored and labeled using the Forager interface. In addition to setting up the dataset, the script runs inference on all images to compute embeddings (currently ImageNet and CLIP) to support search, clustering, and various other operations in Forager.

Before running this script, you need write access to the `foragerml` bucket on Google Cloud Storage. Talk to us if you want to be added.

## Data model

Training and validation images should be stored in separate directories on Google Cloud Storage. The directories must be publicly-accessible on Cloud Storage (i.e., `allUsers` should have "Storage Object Viewer" access). While the directories can be nested arbitrarily deep, each image should ultimately have a unique basename (i.e., the final part of the file path after the last slash). For example, instead of having two images at `gs://path/to/train/A/image.jpg` and `gs://path/to/train/B/image.jpg`, you should place the images at `gs://path/to/train/A.jpg` and `gs://path/to/train/B.jpg`).

## Setup/dependencies

Before running this script, install the following Python requirements:
- aiohttp
- click
- numpy
- Pillow
- tqdm

Also install the following packages using the linked instructions:
- [Pytorch](https://pytorch.org/get-started/locally)
- [CLIP](https://github.com/openai/CLIP#usage)
- [Detectron2](https://detectron2.readthedocs.io/en/latest/tutorials/install.html) (the pre-built wheels are easiest if you're on Linux)

Then, download a pre-trained ResNet-50 model from [here](https://dl.fbaipublicfiles.com/detectron2/ImageNetPretrained/MSRA/R-50.pkl
) and place it in the same folder as this script.

## Running the ingest script

Now you're ready to use this script. Usage is as follows:

```
SERVER_URL=[SERVER URL] python ingest.py [DATASET NAME] [TRAIN SET PATH] [VALIDATION SET PATH] [optional: --resnet_batch_size N]
```

where:
- `SERVER URL` is the url (http://ip:port) for the Forager server; ask us if you need this
- `DATASET NAME` is a unique name for this dataset (alphanumeric + dashes and underscores)
- `TRAIN SET PATH` is a path to the directory on Cloud Storage containing training images, of the form `gs://path/to/train/images`
- `VALIDATION SET PATH` is a path to the directory on Cloud Storage containing validation images, of the form `gs://path/to/val/images`
- `N` is the batch size to use when running the images at full resolution through the ResNet-50 model; it's 1 by default, which is required if the images in your dataset aren't all the same size, but, if they are, feel free to increase it to speed things up (8 works well for us, but it depends on the size of your images and the amount of GPU memory you have)

## Ingesting labels

If you already have labels for your dataset, we provide a script to ingest them into Forager as well. The script currently only supports full image labels (i.e., not bounding boxes, polygons, or pixel-level labels).

Your labels must be stored in a JSON file containing a key-value mapping of image paths to a list of annotations for that image; note that image paths are stripped to the basename (defined above) so it doesn't matter what directory they're relative to. There should be at most one annotation per category per image; behavior otherwise is undefiend. Each annotation can be in any of the following formats:
- a string `"[CATEGORY]"` of which the image is a positive example
- an object `{category: "[CATEGORY]"}` of which the image is a positive example
- an object `{category: "[CATEGORY]", value: "[MODE]"}`. Built-in modes are (case-sensitive, all uppercase) `POSITIVE`, `NEGATIVE`, `HARD_NEGATIVE`, and `UNSURE`, but you can also define custom modes (common ones are `occluded`, `small`, `far away`, etc.).

You must have first ingested the actual dataset using the command above before ingesting labels. To ingest labels, simply run:

```
SERVER_URL=[SERVER URL] python ingest_labels.py [DATASET NAME] [LABEL JSON PATH] --user [YOUR EMAIL]
```

where:
- `SERVER URL` is the url (http://ip:port) for the Forager server; ask us if you need this
- `DATASET NAME` is a unique name for this dataset (alphanumeric + dashes and underscores)
- `LABEL JSON PATH` is a local path to the JSON file containing labels in the format above
- `YOUR EMAIL` is your email address (required for label audit purposes)

## Viewing data

Once ingest is complete, you can view your dataset at `http://35.199.179.109:4000/[DATASET NAME]`.
