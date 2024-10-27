import os
import logging
from typing import Dict, Literal, Tuple, List
from time import time

import cv2
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
from models.load_elevations import load_and_add_osm_data, load_and_add_elevation_data

from models.data_model import BatchDict
from utils import bcolors


logger: logging.Logger = logging.getLogger()  # The logger used to log output


def normalize(img):
    band_min, band_max = (np.min(img, axis=(1, 2)), np.quantile(img, 0.999, axis=(1, 2)))
    div = band_max - band_min
    div[div == 0] = 1 # Avoid division by zero
    return ((img - band_min[:, None, None]) / div[:, None, None]).clip(0, 1)

transform = A.Compose([
    A.HorizontalFlip(p=0.5),  # Горизонтальный поворот
    A.VerticalFlip(p=0.5),    # Вертикальный поворот
    A.RandomBrightnessContrast(p=0.2),  # Изменение яркости и контрастности
    #A.HueSaturationValue(p=0.2),  # Изменение оттенка и насыщенности
    #ToTensorV2()  # Преобразование в тензор PyTorch
])

def augmentations(image, mask):
    augmented = transform(image=image.transpose(1, 2, 0),  # Меняем оси на (H, W, C)
                          mask=mask.transpose(1, 2, 0))
    augmented_image = augmented['image'].transpose(2, 0, 1)
    augmented_mask = augmented['mask'].transpose(2, 0, 1)
    return augmented_image, augmented_mask

class SegmentationDataset(Dataset):
    def __init__(
        self,
        data_path: str,
        data_split: Literal["train", "val", "test"],
        tile_size: int = 256,
        channels: int = 10,
        worldfloods_cnt: int = 0,
        worldfloods_folder: str = "",
        worldfloods_files: List[str] = None,
        include_osm: bool = False,
        include_elevation: bool = False,
        include_dwm: bool = False,
    ) -> None:
        super().__init__()
        self.data_path = data_path
        self.data_split = data_split
        self.tile_size = tile_size
        self.channels = channels
        self.worldfloods_cnt = worldfloods_cnt
        self.worldfloods_folder = worldfloods_folder
        self.worldfloods_files = worldfloods_files
        self.include_osm = include_osm
        self.include_elevation = include_elevation
        self.include_dwm = include_dwm
        if include_elevation and include_osm:
            raise ValueError("Cant add both osm and elevation yet")
        
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
        image = torch.from_numpy(image.copy()).to(torch.float)
        if mask is not None:
            mask = torch.from_numpy(mask.copy()).to(torch.float)
        if self.data_split != "test":
            return dict(
                image=image,
                mask=mask,
                coords=self.coords[index],
                image_index=self.image_indices[index]
            )
        else:
            return dict(
                image=image,
                coords=self.coords[index],
                image_index=self.image_indices[index],
                test_file_path=self.test_file_path[index]
            )

    def _load_data(self) -> None:
        """
        Load train or val data and divide it into small tiles (f.g. 256x256)
        """
        if self.data_split == "train":
            file_names = ['1', '2', '4', '5', '6_1', '6_2']
        elif self.data_split == "val":
            file_names = ['6_1', '6_2', '9_1', '9_2']
        elif self.data_split == "test":
            dir_path = os.path.join(self.data_path, 'images')
            file_names = os.listdir(dir_path)
            for f in file_names:
                assert f.endswith('.tif')
            file_names = sorted(file_names)
            
        if self.worldfloods_cnt > 0 and self.data_split == "train":
            if self.worldfloods_files is None:
                random.seed(42)
                self.worldfloods_files = random.sample(os.listdir(os.path.join(self.worldfloods_folder, "train", "S2")),
                                                       self.worldfloods_cnt)
            file_names += self.worldfloods_files
        
        pictures_and_masks = []
        logger.info(f'uploading {self.data_split} data...')
        for file_name in tqdm(file_names):

            if self.data_split == "test":
                image_path = os.path.join(self.data_path, 'images', file_name)
                with rasterio.open(image_path) as fin:
                    if self.include_elevation:
                        picture = load_and_add_elevation_data(image_path)
                        print(picture.shape)
                    elif self.include_osm:
                        picture = load_and_add_osm_data(image_path)
                    else:
                        picture = fin.read()

                    if self.include_dwm:
                        dwm_path = os.path.join(self.data_path, 'images', "dwm_" + file_name + '.png')
                        dwm_image = cv2.imread(dwm_path, cv2.IMREAD_GRAYSCALE)
                        picture = np.concatenate([picture, np.expand_dims(dwm_image, axis=0)], axis=0)
                    picture = normalize(picture.astype(np.float32))
                pictures_and_masks.append((picture, None))
                continue
            
            if file_name.endswith('.tif'):
                image, mask = load_from_folder(self.worldfloods_folder, self.data_split, file_name, osm=self.include_osm,
                                               elevation=self.include_elevation)
                image = normalize(image.astype(np.float32))
                pictures_and_masks.append((image.astype(np.float32), mask.astype(np.float32)))
                continue
            
            image_path = os.path.join(self.data_path, 'images', file_name + '.tif')
            mask_path = os.path.join(self.data_path, 'masks', file_name + '.tif')
            with rasterio.open(image_path) as fin:
                picture = None
                if self.include_elevation:
                    picture = load_and_add_elevation_data(image_path)
                    print(picture.shape)
                elif self.include_osm:
                    picture = load_and_add_osm_data(image_path)
                else:
                    picture = fin.read()
                picture = normalize(picture.astype(np.float32))
                if self.data_split == "train" and file_name == '6_1':
                    picture = picture[:, :, :8000]
                elif self.data_split == "val" and file_name == '6_1':
                    picture = picture[:, :, 8000:]
                if self.data_split == "train" and file_name == '6_2':
                    picture = picture[:, :, :5000]
                elif self.data_split == "val" and file_name == '6_2':
                    picture = picture[:, :, 5000:]
            with rasterio.open(mask_path) as fin:
                mask = fin.read(1)
                if self.data_split == "train" and file_name == '6_1':
                    mask = mask[:, :8000]
                elif self.data_split == "val" and file_name == '6_1':
                    mask = mask[:, 8000:]
                if self.data_split == "train" and file_name == '6_2':
                    mask = mask[:, :5000]
                elif self.data_split == "val" and file_name == '6_2':
                    mask = mask[:, 5000:]
                mask = np.expand_dims(mask, axis=0)
            if self.channels >= 12:
                dwm_path = os.path.join(self.data_path, 'images', "dwm_" + file_name + '.png')
                dwm_image = cv2.imread(dwm_path, cv2.IMREAD_GRAYSCALE)

                if self.data_split == "train" and file_name == '6_1':
                    dwm_image = dwm_image[:, :8000]
                elif self.data_split == "val" and file_name == '6_1':
                    dwm_image = dwm_image[:, 8000:]
                if self.data_split == "train" and file_name == '6_2':
                    dwm_image = dwm_image[:, :5000]
                elif self.data_split == "val" and file_name == '6_2':
                    dwm_image = dwm_image[:, 5000:]
                picture = np.concatenate([picture, np.expand_dims(dwm_image, axis=0)], axis=0)
            pictures_and_masks.append((picture.astype(np.float32), mask.astype(np.float32)))
        
        logger.info(f'dividing into tiles ...')
        self.images = []
        self.masks = []
        self.coords = []
        self.image_indices = []
        self.test_file_path = []

        tile_size = self.tile_size
        padding_size = tile_size // 2
        
        total_worldfloods_tiles = 0
        total_baseline_tiles = 0
        for image_idx, (image, mask) in enumerate(tqdm(pictures_and_masks)):
            if mask is not None:
                assert image.shape[1:] == mask.shape[1:], (image.shape, mask.shape)
            ch, h, w = image.shape
            h_plus = -1
            w_plus = -1
            if self.data_split == "val" or self.data_split == "test":
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
                    
                    if mask is not None:
                        tile_mask = mask[:, i: i + tile_size, j: j + tile_size]
                    tile_image = image[:, i: i + tile_size, j: j + tile_size]

                    # if self.data_split == "train" and mask.sum() < 0.05 * mask.shape[0] * mask.shape[1]:
                    #     continue

                    if ch > self.channels:
                        # Image from worldfloods
                        if not is_tile_valid(tile_mask):
                            continue
                        tile_image, tile_mask = from_worldfloods(tile_image, tile_mask,
                                                                 elevation=self.include_osm or self.include_elevation)
                        total_worldfloods_tiles += 1
                    else:
                        total_baseline_tiles += 1
                    
                    self.images.append(tile_image)
                    if mask is not None:
                        self.masks.append(tile_mask)
                    else:
                        self.masks.append(None)
                    self.coords.append((i, j))
                    self.image_indices.append(image_idx)
                    if self.data_split == "test":
                        self.test_file_path.append(file_names[image_idx])
        
        logger.info(f'Baseline tiles for {self.data_split}: {total_baseline_tiles}')
        logger.info(f'Worldfloods tiles for {self.data_split}: {total_worldfloods_tiles}')


def get_dataloader(config: Dict, data_split: str, batch_size: int) -> DataLoader:
    return DataLoader(
        SegmentationDataset(
            config[data_split]['dataset_path'], data_split,
            config["train_params"]['tile_size'],
            channels=config["train_model"]["num_channels"],
            worldfloods_cnt=config.get("worldfloods_cnt", 0),
            worldfloods_folder=config.get("worldfloods_folder", ""),
            include_osm=config.get("include_osm", False),
            include_elevation=config.get("include_elevation", False),
            include_dwm=config.get("include_dwm", False)
            ),
        batch_size=batch_size,
        num_workers=config[data_split]["num_workers"],
        shuffle=config[data_split]["shuffle"],
        drop_last=config[data_split]["drop_last"],
    )
