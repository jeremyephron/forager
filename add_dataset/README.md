# Adding a dataset to Forager

This script allows you to ingest data into a running Forager instance. Specifically, it turns directories of training and validation images on [Google Cloud Storage](https://cloud.google.com/storage) into a Forager "dataset" that can be explored and labeled using the Forager interface. In addition to setting up the dataset, the script runs inference on all images to compute embeddings to support search, clustering, and various other operations in Forager. (It currently generates ImageNet and CLIP embeddings.)

Before running this script, you need write access to the `foragerml` bucket on Google Cloud Storage. Talk to us if you want to be added.

## Data model

Training and validation images should be stored in separate directories on Google Cloud Storage. The directories must be publicly-accessible on Cloud Storage (i.e., `allUsers` should have "Storage Object Viewer" access: see [here](https://cloud.google.com/storage/docs/access-control/making-data-public) for more info). The contents of these directories can be nested arbitrarily deep -- the ingest script will recurse into subdirectories to find all images, however each image should ultimately have a unique basename (i.e., the final part of the file path after the last slash). For example, instead of having two images at `gs://path/to/train/A/image.jpg` and `gs://path/to/train/B/image.jpg`, you should place the images at `gs://path/to/train/A.jpg` and `gs://path/to/train/B.jpg`).

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
SERVER_URL=[SERVER URL] python main.py [DATASET NAME] [TRAIN SET PATH] [VALIDATION SET PATH] [optional: --resnet_batch_size N]
```

where:
- `SERVER URL` is the url (http://ip:port) for the Forager server; ask us if you need this
- `DATASET NAME` is a unique name for this dataset (alphanumeric + dashes and underscores)
- `TRAIN SET PATH` is a path to the directory on Google Cloud Storage containing training images, of the form `gs://path/to/train/images`
- `VALIDATION SET PATH` is a path to the directory on Google Cloud Storage containing validation images, of the form `gs://path/to/val/images`
- `N` is the batch size to use when performing inference using the ResNet-50 model; it's 1 by default, which is required if the images in your dataset aren't all the same size (full resolution images are sent through the model). If you know images in your collection are all the same size, feel free to increase the batch size to increase inference performance. (8 works well for us, but it depends on the size of your images and the amount of GPU memory you have.)

## Viewing data

Once ingest is complete, you can view your dataset at `http://35.199.179.109:4000/[DATASET NAME]`.
