import numpy as np


class EarlyStopper:
    def __init__(self, patience: int = 1, delta: int = 0):
        self.patience = patience
        self.delta = delta
        self.counter = 0
        self.max_validation_metric = -np.inf

    def __call__(self, validation_metric) -> bool:
        if validation_metric > self.max_validation_metric:
            self.max_validation_metric = validation_metric
            self.counter = 0
        elif validation_metric <= (self.max_validation_metric - self.delta):
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False
