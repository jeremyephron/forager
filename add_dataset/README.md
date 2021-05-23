# Adding a dataset to Forager

Forager currently supports adding image datasets from Google Cloud Storage. Training and validation images should be stored in separate directories on Cloud Storage; each of these directories can be nested arbitrarily deep.

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
) and place it in this folder.

Now you're ready to use this script. Usage is as follows:

```
python main.py [DATASET NAME] [TRAIN SET PATH] [VALIDATION SET PATH] [optional: --resnet_batch_size N]
```

where:
- `DATASET NAME` is a unique name for this dataset (alphanumeric + dashes and underscores)
- `TRAIN SET PATH` is a path to the directory on Cloud Storage containing training images, of the form gs://path/to/train/images
- `VALIDATION SET PATH` is a path to the directory on Cloud Storage containing validation images, of the form gs://path/to/val/images
- `N` is the batch size to use when running the images at full resolution through the ResNet-50 model; it's 1 by default, which is required if the images in your dataset aren't all the same size, but, if they are, feel free to increase it to speed things up (8 works well for us, but it depends on the size of your images and the amount of GPU memory you have)
