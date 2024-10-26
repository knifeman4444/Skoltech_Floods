import os
from typing import Literal

import cv2
import numpy as np
import rasterio
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from models.data_loader import normalize


def transform_arr(arr):
    """From torch format to pyplot format"""
    return np.transpose(arr[::-1], (1, 2, 0))


def load_raster(src):
    """Load a raster image"""
    data = src.read([1, 2, 3])
    return transform_arr(data / 1500)


def load_mask(path):
    """Load a mask"""
    with rasterio.open(path) as src:
        data = src.read(1)
        return data


class VisualisationDataset(Dataset):
    def __init__(
            self,
            data_path: str,
            image_name: str,
            tile_size: int = 256,
            channels: int = 10,
    ) -> None:
        super().__init__()
        self.data_path = data_path
        self.image_name = image_name
        self.tile_size = tile_size
        self.channels = channels
        self.batch_size = None

        self._load_data()

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, index: int):
        image = self.images[index]
        mask = self.masks[index]

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

        pictures_and_masks = []
        image_path = os.path.join(self.data_path, 'images', self.image_name + '.tif')
        mask_path = os.path.join(self.data_path, 'masks', self.image_name + '.tif')
        with rasterio.open(image_path) as fin:
            picture = []
            for i in range(self.channels):
                chan = normalize(fin.read(i + 1))
                picture.append(chan)
            picture = np.stack(picture)
        with rasterio.open(mask_path) as fin:
            mask = fin.read(1)
            mask = np.expand_dims(mask, axis=0)
        pictures_and_masks.append((picture.astype(np.float32), mask.astype(np.float32)))

        self.images = []
        self.masks = []
        tile_size = self.tile_size

        for image, mask in pictures_and_masks:
            assert image.shape[1:] == mask.shape[1:], (image.shape, mask.shape)
            ch, h, w = image.shape

            for h_coord in range(0, h // tile_size):
                for w_coord in range(0, w // tile_size):
                    y = h_coord * tile_size
                    x = w_coord * tile_size

                    tile_mask = mask[:, y: y + tile_size, x: x + tile_size]
                    tile_image = image[:, y: y + tile_size, x: x + tile_size]

                    self.images.append(tile_image)
                    self.masks.append(tile_mask)
            self.batch_size = w // tile_size


def visualize_model_predictions_for_image(model, config, image_name):
    data_split = 'train'
    ds = VisualisationDataset(
        '../dataset/train',
        image_name,
        config["train_params"]['tile_size'],
    )
    dataloader = DataLoader(
        dataset=ds,
        batch_size=ds.batch_size,
        num_workers=config[data_split]["num_workers"],
        shuffle=False,
        drop_last=False,
    )
    rows = []
    for step, batch in enumerate(dataloader):  # Receive a row
        preds = model.forward(batch["image"].to(config["device"])).to("cpu")
        mask = batch["mask"]
        aa = np.concatenate(
            [
                preds,
                np.zeros(preds.shape),
                mask
            ],
            axis=1
        )
        img = batch["image"][:,:3,:,:].numpy()
        row_result = np.concatenate(255 * (0.8 * aa + 0.2 * img * 10), axis=2)
        rows.append(np.transpose(row_result, (1, 2, 0)))
    result = np.vstack(rows)
    path_to_file = config['val']['output_dir']
    cv2.imwrite(f'{path_to_file}/{image_name}.jpg', result)


def visualize_model_predictions(model, config):
    model.eval()
    with torch.no_grad():
        visualize_model_predictions_for_image(model, config, '9_1')
        visualize_model_predictions_for_image(model, config, '9_2')
