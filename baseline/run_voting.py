import os
import numpy as np
import rasterio
from rasterio.enums import Resampling
from utils import initialize_logging, load_config
from tqdm import tqdm


def load_masks_from_folders_by_name(folder_paths):
    mask_groups = {}
    meta_groups = {}
    for folder in folder_paths:
        files = [f for f in os.listdir(folder) if f.endswith('.tif')]
        files.sort()
        for file_name in files:
            file_path = os.path.join(folder, file_name)
            with rasterio.open(file_path) as src:
                meta = src.meta
                if file_name not in mask_groups:
                    mask_groups[file_name] = []
                mask_groups[file_name].append(src.read(1))  # Assuming single-channel mask
                meta_groups[file_name] = meta
    return mask_groups, meta_groups

def majority_voting(masks):
    # Majority voting method
    stacked_masks = np.stack(masks, axis=0)
    return (stacked_masks.sum(axis=0) >= stacked_masks.shape[0] // 2 + 1).astype(float)

def process_masks(input_folders, output_folder):
    print('Processing masks...')
    mask_groups, meta_groups = load_masks_from_folders_by_name(input_folders)

    print('Processing majority_voting and saving...')
    os.makedirs(output_folder, exist_ok=True)
    for file_name, masks in tqdm(mask_groups.items()):
        assert len(masks) > 1
        majority_mask = majority_voting(masks)
        
        with rasterio.open(os.path.join(output_folder, file_name), 'w', **meta_groups[file_name]) as fout:
            fout.write(majority_mask, 1)


# Настройте пути к вашим папкам
config = load_config(config_path="./config/local_config.yaml")
prefix = os.path.join(config['root_path'], 'dataset/train')

input_folders = ['preds_-', 'preds_-', 'preds_-']
output_folder = '-'.join(input_folders)

input_folders = [os.path.join(prefix, folder) for folder in input_folders]
output_folder = os.path.join(prefix, output_folder)
process_masks(input_folders, output_folder)
