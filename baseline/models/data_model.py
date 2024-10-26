from typing import TypedDict

import torch


class BatchDict(TypedDict):
    image: torch.Tensor
    mask: torch.Tensor


class Postfix(TypedDict):
    Epoch: int
    train_loss: float
    train_loss_step: float
    f1: float
    iou: float


class TestResults(TypedDict):
    f1: float
    iou: float
