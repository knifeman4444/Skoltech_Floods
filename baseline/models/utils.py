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

from sklearn.metrics import f1_score


def reduce_func(D_chunk, start):
    top_size = 100
    nearest_items = np.argsort(D_chunk, axis=1)[:, :top_size + 1]
    return [(i, items[items!=i]) for i, items in enumerate(nearest_items, start)]

def calculate_metrics(y_true, y_pred) -> Dict[str, float]:
    # Calculate intersection and union
    intersection = np.sum((y_true == 1) & (y_pred == 1))
    union = np.sum((y_true == 1) | (y_pred == 1))
    epsilon=1e-7

    # Calculate IoU
    iou = (intersection + epsilon) / (union + epsilon)
    f1 = f1_score(y_true, y_pred, average='macro')

    return {'f1': f1, 'iou': iou}


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

def save_predictions(outputs: Dict[str, np.ndarray], output_dir: str) -> None:
    #TODO
    os.makedirs(output_dir, exist_ok=True)
    for key in outputs:
        if "_ids" in key:
            with jsonlines.open(os.path.join(output_dir, f"{key}.jsonl"), "w") as f:
                if len(outputs[key][0]) == 4:
                    for clique, anchor, pos, neg in outputs[key]:
                        f.write({"clique_id": clique, "anchor_id": anchor, "positive_id": pos, "negative_id": neg})
                else:
                    for clique, anchor in outputs[key]:
                        f.write({"clique_id": clique, "anchor_id": anchor})
        else:
            np.save(os.path.join(output_dir, f"{key}.npy"), outputs[key])


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
