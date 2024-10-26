import os
import logging
from typing import Dict, Literal, Tuple
from time import time
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import rasterio
import random
import albumentations as A
from albumentations.pytorch import ToTensorV2
from tqdm import tqdm

from models.data_model import BatchDict
from utils import bcolors


logger: logging.Logger = logging.getLogger()  # The logger used to log output


def normalize(band):
    band_min, band_max = (band.min(), band.max())
    return ((band - band_min) / ((band_max - band_min)))


transform = A.Compose([
    A.HorizontalFlip(p=0.5),  # Горизонтальный поворот
    A.VerticalFlip(p=0.5),    # Вертикальный поворот
    #A.RandomBrightnessContrast(p=0.2),  # Изменение яркости и контрастности
    #A.HueSaturationValue(p=0.2),  # Изменение оттенка и насыщенности
    #ToTensorV2()  # Преобразование в тензор PyTorch
])

def augmentations(image, mask):
    augmented = transform(image=image.transpose(1, 2, 0),  # Меняем оси на (H, W, C)
                          mask=mask.transpose(1, 2, 0))
    augmented_image = augmented['image'].transpose(2, 0, 1)
    augmented_mask = augmented['mask'].transpose(2, 0, 1)
    return augmented_image, augmented_mask


class CoverDataset(Dataset):
    def __init__(
        self,
        data_path: str,
        data_split: Literal["train", "val"],
        tile_size: int = 256
    ) -> None:
        super().__init__()
        self.data_path = data_path
        self.data_split = data_split
        self.tile_size = tile_size
        self._load_data()

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, index: int):
        image = self.images[index]
        mask = self.masks[index]
        if self.data_split == "train":
            image, mask = augmentations(image, mask)
        image = torch.from_numpy(image.copy())
        mask = torch.from_numpy(mask.copy())
        return dict(
            image=image,
            mask=mask,
        )

    def _load_data(self) -> None:
        """
        Load train or val data and divide it into small tiles (f.g. 256x256)
        """
        if self.data_split == "train":
            file_names = ['1', '2', '4', '5', '6_1', '6_2']
        elif self.data_split == "val":
            file_names = ['6_1', '6_2', '9_1', '9_2']
        
        pictures_and_masks = []
        logger.info(f'uploading {self.data_split} data...')
        for file_name in tqdm(file_names[:1]):
            image_path = os.path.join(self.data_path, 'images', file_name + '.tif')
            mask_path = os.path.join(self.data_path, 'masks', file_name + '.tif')
            with rasterio.open(image_path) as fin:
                picture = []
                for i in range(10):
                    chan = normalize(fin.read(i + 1))
                    if self.data_split == "train" and file_name[0] == '6':
                        chan = chan[:, :8000]
                    elif self.data_split == "val" and file_name[0] == '6':
                        chan = chan[:, 8000:]
                    picture.append(chan)
                picture = np.stack(picture)
            with rasterio.open(mask_path) as fin:
                mask = fin.read(1)
                if self.data_split == "train" and file_name[0] == '6':
                    mask = mask[:, :8000]
                elif self.data_split == "val" and file_name[0] == '6':
                    mask = mask[:, 8000:]
                mask = np.expand_dims(mask, axis=0)
            pictures_and_masks.append((picture.astype(np.float32), mask.astype(np.float32)))
        
        logger.info(f'dividing into tiles ...')
        self.images = []
        self.masks = []
        tile_size = self.tile_size
        for image, mask in tqdm(pictures_and_masks):
            assert image.shape[1:] == mask.shape[1:], (image.shape, mask.shape)
            _, h, w = image.shape

            for h_coord in range(0, h // tile_size):
                for w_coord in range(0, w // tile_size):
                    y = h_coord * tile_size
                    x = w_coord * tile_size
                    self.images.append(image[:, y: y + tile_size, x: x + tile_size])
                    self.masks.append(mask[:, y: y + tile_size, x: x + tile_size])


def get_dataloader(config: Dict, data_split: str, batch_size: int) -> DataLoader:
    return DataLoader(
        CoverDataset(config[data_split]['dataset_path'], data_split, config["train_params"]['tile_size']),
        batch_size=batch_size,
        num_workers=config[data_split]["num_workers"],
        shuffle=config[data_split]["shuffle"],
        drop_last=config[data_split]["drop_last"],
    )
