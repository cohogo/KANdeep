import sys
import glob
from os.path import join
import numpy as np
import os
from PIL import Image, ImageEnhance
import torch
from torch.utils.data import Dataset, DataLoader, TensorDataset
import torchvision.transforms as T
import importlib
config_module = os.environ.get("DEEPMIH_CONFIG", "config")
c = importlib.import_module(config_module)
from natsort import natsorted

def _collect_files(path: str, pattern: str, mode: str):
    if not os.path.isdir(path):
        raise FileNotFoundError(
            f"Dataset path for {mode} does not exist: {path}. "
            "Update config.py to point to the extracted images."
        )
    files = natsorted(sorted(glob.glob(join(path, f"*.{pattern}"))))
    if not files:
        raise RuntimeError(
            f"Dataset path for {mode} is empty: {path}. "
            "Ensure the directory contains *.{pattern} images."
        )
    return files

def to_rgb(image):
    rgb_image = Image.new("RGB", image.size)
    rgb_image.paste(image)
    return rgb_image


class Hinet_Dataset(Dataset):
    def __init__(self, transforms_=None, mode="train"):

        self.transform = transforms_
        self.mode = mode
        if self.mode == "train":
            # TRAIN SETTING
            if c.Dataset_mode == 'DIV2K':
                path = c.TRAIN_PATH_DIV2K
                pattern = 'png'
                print('TRAIN DATASET is DIV2K')

            elif c.Dataset_mode == 'COCO':
                path = c.TEST_PATH_COCO
                pattern = 'jpg'
                print('TRAIN DATASET is COCO')
            else:
                raise ValueError(f"Unsupported Dataset_mode: {c.Dataset_mode}")

            # train
            self.files = _collect_files(path, pattern, mode="train")

        elif self.mode == "val":
            # VAL SETTING
            if c.Dataset_VAL_mode == 'DIV2K':
                self.VAL_PATH = c.VAL_PATH_DIV2K
                self.format_val = 'png'
                print('VAL DATASET is DIV2K')

            elif c.Dataset_VAL_mode == 'COCO':
                path = c.VAL_PATH_COCO
                pattern = 'jpg'
                print('VAL DATASET is COCO')

            elif c.Dataset_VAL_mode == 'ImageNet':
                path = c.VAL_PATH_IMAGENET
                pattern = 'JPEG'
                print('VAL DATASET is ImageNet')
            else:
                raise ValueError(f"Unsupported Dataset_VAL_mode: {c.Dataset_VAL_mode}")

            # test
            self.files = _collect_files(path, pattern, mode="val")

        else:
            raise ValueError(f"Unsupported dataset mode: {self.mode}")

    def __getitem__(self, index):
        if not self.files:
            raise RuntimeError(f"No files available for mode={self.mode}. Check dataset paths in config.py.")

        length = len(self.files)
        index = index % length

        for offset in range(length):
            real_index = (index + offset) % length
            path = self.files[real_index]
            try:
                with Image.open(path) as img:
                    image = to_rgb(img)
                if self.transform is not None:
                    return self.transform(image)
                return image
            except Exception as exc:
                print(f"[WARN] Failed to load {path}: {exc}. Trying next file.")

        raise RuntimeError(
            "All files in dataset failed to load. "
            "Ensure the dataset does not contain only corrupt images."
        )

    def __len__(self):
        return len(self.files)


if c.Dataset_VAL_mode == 'DIV2K':
    cropsize_val = c.cropsize_val_div2k
if c.Dataset_VAL_mode == 'COCO':
    cropsize_val = c.cropsize_val_coco
if c.Dataset_VAL_mode == 'ImageNet':
    cropsize_val = c.cropsize_val_imagenet

transform = T.Compose([
    T.RandomHorizontalFlip(),
    T.RandomVerticalFlip(),
    T.RandomCrop(c.cropsize),
    T.ToTensor()
])

transform_val = T.Compose([
    T.CenterCrop(cropsize_val),
    T.ToTensor(),
])


# Training data loader
trainloader = DataLoader(
    Hinet_Dataset(transforms_=transform, mode="train"),
    batch_size=c.batch_size,
    shuffle=True,
    pin_memory=False,
    num_workers=1,
    drop_last=True
)
# Test data loader
testloader = DataLoader(
    Hinet_Dataset(transforms_=transform_val, mode="val"),
    batch_size=c.batchsize_val,
    shuffle=False,
    pin_memory=False,
    num_workers=1,
    drop_last=True
)