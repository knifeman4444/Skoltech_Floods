from typing import TypedDict, Tuple

import torch


class BatchDict(TypedDict):
    image: torch.Tensor
    mask: torch.Tensor
    coords: Tuple[int, int]
    image_index: int


class Postfix(TypedDict):
    Epoch: int
    train_loss: float
    train_loss_step: float
    total_f1: float
    f1_water: float
    avg_f1_business: float
    pre_f1: float
    post_f1: float
    f1: float
    iou: float


class TestResults(TypedDict):
    f1: float
    iou: float
