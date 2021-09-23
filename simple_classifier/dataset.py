import glob
import json
from pathlib import Path
from typing import Optional, Union

import torch
from PIL import Image
from torch.utils.data import Dataset


class SimpleDataset(Dataset):
    def __init__(self, data_path: Optional[Union[Path, str]] = None,
                 annotations_path: Optional[Union[str, Path]] = None,
                 transforms=None):
        if isinstance(data_path, str):
            data_path = Path(data_path)
        if isinstance(annotations_path, str):
            annotations_path = Path(annotations_path)
        self.images = glob.glob(str(data_path / "*"))
        self.classes = None
        if annotations_path is not None:
            with open(annotations_path, "r") as file:
                self.classes = json.load(file)
        self.transforms = transforms

    def __len__(self):
        return len(self.images)

    def __getitem__(self, item):
        image = Image.open(self.images[item])

        if self.transforms:
            image = self.transforms(image)

        if self.classes is None:
            return image, self.images[item]
        annot = self.classes[self.images[item]]
        return image, torch.Tensor(annot)
