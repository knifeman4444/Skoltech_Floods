import os
import logging
from typing import Dict, Literal, Tuple, List
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
from models.dataset_converter import is_tile_valid, load_from_folder, from_worldfloods

from models.data_model import BatchDict
from utils import bcolors


logger: logging.Logger = logging.getLogger()  # The logger used to log output


def normalize(band):
    band_min, band_max = (band.min(), band.max())
    if band_max == band_min:
        return np.zeros(band.shape)
    return (band - band_min) / (band_max - band_min)


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
        tile_size: int = 256,
        channels: int = 10,
        worldfloods_cnt: int = 0,
        worldfloods_folder: str = "",
        worldfloods_files: List[str] = None
    ) -> None:
        super().__init__()
        self.data_path = data_path
        self.data_split = data_split
        self.tile_size = tile_size
        self.channels = channels
        self.worldfloods_cnt = worldfloods_cnt
        self.worldfloods_folder = worldfloods_folder
        self.worldfloods_files = worldfloods_files
        
        if worldfloods_cnt > 0 and not os.path.exists(worldfloods_folder):
            raise FileNotFoundError(f"Folder {worldfloods_folder} does not exist")
        if worldfloods_cnt > 0 and worldfloods_files is not None and len(worldfloods_files) != worldfloods_cnt:
            raise ValueError(f"Mismatch in number of files: {len(worldfloods_files)} != {worldfloods_cnt}")
        
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
            coords=self.coords[index],
            image_index=self.image_indices[index]
        )

    def _load_data(self) -> None:
        """
        Load train or val data and divide it into small tiles (f.g. 256x256)
        """
        if self.data_split == "train":
            file_names = ['1', '2', '4', '5', '6_1', '6_2']
        elif self.data_split == "val":
            file_names = ['6_1', '6_2', '9_1', '9_2']
            
        if self.worldfloods_cnt > 0 and self.data_split == "train":
            if self.worldfloods_files is None:
                random.seed(42)
                self.worldfloods_files = random.sample(os.listdir(os.path.join(self.worldfloods_folder, "test", "S2")),
                                                       self.worldfloods_cnt)
            file_names += self.worldfloods_files
        
        pictures_and_masks = []
        logger.info(f'uploading {self.data_split} data...')
        for file_name in tqdm(file_names):
            
            if not file_name[0].isdigit():
                image, mask = load_from_folder(self.worldfloods_folder, 'test', file_name)
                for i in range(image.shape[0]):
                    image[i] = normalize(image[i])
                pictures_and_masks.append((image.astype(np.float32), mask.astype(np.float32)))
                continue
            
            image_path = os.path.join(self.data_path, 'images', file_name + '.tif')
            mask_path = os.path.join(self.data_path, 'masks', file_name + '.tif')
            with rasterio.open(image_path) as fin:
                picture = []
                for i in range(self.channels):
                    chan = normalize(fin.read(i + 1))
                    if self.data_split == "train" and file_name[0] == '6_1':
                        chan = chan[:, :8000]
                    elif self.data_split == "val" and file_name[0] == '6_1':
                        chan = chan[:, 8000:]
                    if self.data_split == "train" and file_name[0] == '6_2':
                        chan = chan[:, :5000]
                    elif self.data_split == "val" and file_name[0] == '6_2':
                        chan = chan[:, 5000:]
                    picture.append(chan)
                picture = np.stack(picture)
            with rasterio.open(mask_path) as fin:
                mask = fin.read(1)
                if self.data_split == "train" and file_name[0] == '6_1':
                    mask = mask[:, :8000]
                elif self.data_split == "val" and file_name[0] == '6_1':
                    mask = mask[:, 8000:]
                if self.data_split == "train" and file_name[0] == '6_2':
                    mask = mask[:, :5000]
                elif self.data_split == "val" and file_name[0] == '6_2':
                    mask = mask[:, 5000:]
                mask = np.expand_dims(mask, axis=0)
            pictures_and_masks.append((picture.astype(np.float32), mask.astype(np.float32)))
        
        logger.info(f'dividing into tiles ...')
        self.images = []
        self.masks = []
        self.coords = []
        self.image_indices = []

        tile_size = self.tile_size
        padding_size = tile_size // 2
        
        total_worldfloods_tiles = 0
        total_baseline_tiles = 0
        for image_idx, (image, mask) in enumerate(tqdm(pictures_and_masks)):
            assert image.shape[1:] == mask.shape[1:], (image.shape, mask.shape)
            ch, h, w = image.shape
            h_plus = -1
            w_plus = -1
            if self.data_split == "val":
                if h % padding_size != 0:
                    h_plus = 0
                if w % padding_size != 0:
                    w_plus = 0

            for h_coord in range(0, h // padding_size + h_plus):
                for w_coord in range(0, w // padding_size + w_plus):
                    i = h_coord * padding_size
                    j = w_coord * padding_size
                    i = min(i, h - tile_size)
                    j = min(j, w - tile_size)
                    
                    tile_mask = mask[:, i: i + tile_size, j: j + tile_size]
                    tile_image = image[:, i: i + tile_size, j: j + tile_size]
                    if ch > 11:
                        # Image from worldfloods
                        if not is_tile_valid(tile_mask):
                            continue
                        tile_image, tile_mask = from_worldfloods(tile_image, tile_mask)
                        total_worldfloods_tiles += 1
                    else:
                        total_baseline_tiles += 1
                    
                    self.images.append(tile_image)
                    self.masks.append(tile_mask)
                    self.coords.append((i, j))
                    self.image_indices.append(image_idx)
        
        logger.info(f'Baseline tiles for {self.data_split}: {total_baseline_tiles}')
        logger.info(f'Worldfloods tiles for {self.data_split}: {total_worldfloods_tiles}')


def get_dataloader(config: Dict, data_split: str, batch_size: int) -> DataLoader:
    return DataLoader(
        CoverDataset(
            config[data_split]['dataset_path'], data_split,
            config["train_params"]['tile_size'],
            worldfloods_cnt=config.get("worldfloods_cnt", 0),
            worldfloods_folder=config.get("worldfloods_folder", ""),
            ),
        batch_size=batch_size,
        num_workers=config[data_split]["num_workers"],
        shuffle=config[data_split]["shuffle"],
        drop_last=config[data_split]["drop_last"],
    )
