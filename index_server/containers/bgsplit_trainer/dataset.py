import torch
import os.path

from torch.utils.data import Dataset
from torchvision import transforms, utils, io
from typing import Dict, List

import config
import util

class AuxiliaryDataset(Dataset):
    def __init__(self, positive_paths: List[str], negative_paths: List[str],
                 unlabeled_paths: List[str],
                 auxiliary_labels: Dict[str, int], transform=None):
        self.paths = positive_paths + negative_paths + unlabeled_paths
        self.main_labels = torch.tensor(
            ([1] * len(positive_paths) +
             [0] * len(negative_paths) +
             [-1] * len(unlabeled_paths)),
            dtype=torch.int)
        self.aux_labels = torch.tensor(
            [auxiliary_labels[os.path.basename(path)]
             if path in auxiliary_labels else -1
             for path in positive_paths + negative_paths + unlabeled_paths],
            dtype=torch.int)
        self.transform = transform

    @property
    def num_auxiliary_classes(self):
        return len(torch.unique(self.aux_labels))

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        path = self.paths[index]
        main_label = self.main_labels[index]
        aux_label = self.aux_labels[index]
        if path.startswith('http'):
            data = util.download(path)
            image = io.decode_image(data, mode=io.ImageReadMode.ImageReadMode.RGB)
        else:
            image = io.read_image(path, mode=io.ImageReadMode.ImageReadMode.RGB)
        image = self.transform(image) if self.transform else image
        return image, main_label, aux_label
