import glob
import json
import os
import shutil
import re
from typing import Dict, List, Tuple

import jsonlines
import numpy as np
import pandas as pd
from sklearn.metrics import pairwise_distances
from torch.utils.data import DataLoader

from models.data_loader import get_dataloader
from models.data_model import Postfix
from calculate_metrics import get_score
import matplotlib.pyplot as plt

from sklearn.metrics import f1_score


def reduce_func(D_chunk, start):
    top_size = 100
    nearest_items = np.argsort(D_chunk, axis=1)[:, :top_size + 1]
    return [(i, items[items!=i]) for i, items in enumerate(nearest_items, start)]


def make_full_masks(y_true, y_pred, coords, indices, tile_size) -> List[Tuple[np.ndarray, np.ndarray]]:
    # Find the number of images
    num_img = indices[-1][-1] + 1
    img_sizes = [[0, 0] for _ in range(num_img)]
    
    # Find the maximum size of the full image H and W
    for i in range(len(coords)):
        for j in range(len(coords[i])):  # Use len() for variable length
            x, y = coords[i][j]
            image_index = indices[i][j]
            img_sizes[image_index][0] = max(img_sizes[image_index][0], x + tile_size)
            img_sizes[image_index][1] = max(img_sizes[image_index][1], y + tile_size)
    
    half_ts = tile_size // 2
    quat_ts = tile_size // 4
    masks = []
    for idx in range(num_img):
        H, W = img_sizes[idx]
        true_full = np.zeros((H, W))
        pred1_full = np.zeros((H, W))
        pred2_full = np.zeros((H, W))

        for i in range(len(y_true)):
            for j in range(len(y_true[i])):
                if indices[i][j] != idx:
                    continue
                x, y = coords[i][j]
                
                # Mask from y_true
                true_full[x:x + tile_size, y:y + tile_size] = y_true[i][j]
                
                # Fill the first predicted full image
                pred1_full[x:x + tile_size, y:y + tile_size] = y_pred[i][j]
                
                # Center for the second predicted full image
                pred2_full[x + quat_ts:x + quat_ts + half_ts, y + quat_ts:y + quat_ts + half_ts] = y_pred[i][j][quat_ts:-quat_ts, quat_ts:-quat_ts]

        pred1_full[quat_ts:-quat_ts, quat_ts:-quat_ts] = pred2_full[quat_ts:-quat_ts, quat_ts:-quat_ts]
        masks.append((true_full.copy(), pred1_full.copy()))
    
    return masks


def calculate_metrics(y_true, y_pred, coords, indices, tile_size, osm_path) -> Tuple[Dict[str, float], List[Tuple[np.ndarray, np.ndarray]]]:
    masks = make_full_masks(y_true, y_pred, coords, indices, tile_size)
    metrics = get_score(masks.copy(), osm_path)

    y_true = np.concatenate([mask[0].flatten() for mask in masks])
    y_pred = np.concatenate([mask[1].flatten() for mask in masks])

    # Calculate intersection and union
    intersection = np.sum((y_true == 1) & (y_pred == 1))
    union = np.sum((y_true == 1) | (y_pred == 1))
    epsilon=1e-7

    # Calculate IoU
    iou = (intersection + epsilon) / (union + epsilon)
    f1 = f1_score(y_true, y_pred, average='macro')

    return {**metrics, 'f1': f1, 'iou': iou}, masks


def dir_checker(output_dir: str, config_path: str) -> str:
    output_dir = re.sub(r"run-[0-9]+/*", "", output_dir)
    runs = glob.glob(os.path.join(output_dir, "run-*"))
    if runs != []:
        max_run = max(map(lambda x: int(x.split("-")[-1]), runs))
        run = max_run + 1
    else:
        run = 0
    outdir = os.path.join(output_dir, f"run-{run}")

    # Create directory for current run and copy config file
    os.makedirs(outdir, exist_ok=True)
    filename = os.path.basename(config_path)
    dst = os.path.join(outdir, filename)
    shutil.copy2(config_path, dst)
    return outdir

def save_test_predictions(predictions: List, output_dir: str) -> None:
    #TODO
    with open(os.path.join(output_dir, 'submission.txt'), 'w') as foutput:
        for query_item, query_nearest in predictions:
            foutput.write('{}\t{}\n'.format(query_item, '\t'.join(map(str,query_nearest))))

def save_predictions(masks: List[Tuple[np.ndarray, np.ndarray]], output_dir: str) -> None:
    os.makedirs(output_dir, exist_ok=True)

    for i, (mask_true, mask_pred) in enumerate(masks):
        plt.figure()  # Создаём новую фигуру для каждого изображения
        
        # Сопоставление изображения и маски
        plt.subplot(1, 2, 1)
        plt.imshow(mask_true, cmap='gray')
        plt.title('True')
        plt.axis('off')
        
        plt.subplot(1, 2, 2)
        plt.imshow(mask_pred, cmap='gray')
        plt.title('Prediction')
        plt.axis('off')
        
        # Сохранение графика в файл
        plt.savefig(os.path.join(output_dir, f"prediction_{i}.png"), bbox_inches='tight')
        plt.close()  # Закрываем фигуру для освобождения памяти


def save_logs(outputs: dict, output_dir: str, name: str = "log") -> None:
    os.makedirs(output_dir, exist_ok=True)
    log_file = os.path.join(output_dir, f"{name}.jsonl")
    with jsonlines.open(log_file, "a") as f:
        f.write(outputs)


def save_best_log(outputs: Postfix, output_dir: str) -> None:
    os.makedirs(output_dir, exist_ok=True)
    log_file = os.path.join(output_dir, "best-log.json")
    with open(log_file, "w") as f:
        json.dump(outputs, f, indent=2)
