import logging
import os
from copy import deepcopy
from typing import Dict, List

import numpy as np
from sklearn.metrics import pairwise_distances_chunked
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm, trange

import models.custom_models
from models.data_model import BatchDict, Postfix, TestResults
from models.early_stopper import EarlyStopper
from models.utils import (
    calculate_metrics,
    dir_checker,
    make_full_masks,
    save_best_log,
    save_logs,
    save_predictions,
    save_test_predictions
)
from models.data_loader import get_dataloader
import segmentation_models_pytorch as smp
import wandb

from models.visualize import visualize_model_predictions

logger: logging.Logger = logging.getLogger()  # The logger used to log output


def load_model(config):
    if not config["train_model"].get('model'):
        return smp.Unet(
            encoder_name=config["train_model"]["backbone"],
            encoder_weights=config["train_model"]["pretrain"],
            in_channels=config["train_model"]["num_channels"],
            classes=1,
            activation='sigmoid'
        )
    elif config["train_model"]['model'] == "CustomVIT":
        return models.custom_models.CustomVIT(config=config)
    else:
        raise RuntimeError(f"model {config['train_model']['model']} in config does not exist")


class TrainModule:
    def __init__(self, config: Dict, wandb_token: str, config_path: str) -> None:
        # Initialize the parameters from the config
        self.config = config
        self.state = "initializing"
        self.best_model_path: str = None
        self.wandb_log = self.config["wandb_log"]
        self.config_path = config_path

        self.model = load_model(config)
        #self.model.to(self.config["device"])
        self.model = nn.DataParallel(self.model).to(self.config["device"])
        self.postfix: Postfix = {}

        def my_dice_loss(p, y):
            loss = 1 - (2 * (p * y).sum() + 1) / (p.sum() + y.sum() + 1)
            return loss
        
        self.loss = my_dice_loss
        self.early_stop = EarlyStopper(patience=self.config["train"]["patience"])
        self.optimizer = self.configure_optimizers()
        self.wandb_token = wandb_token

        self.config["train"]["dataset_path"] = os.path.join(self.config["root_path"], self.config["train"]["dataset_path"])
        self.config["val"]["dataset_path"] = os.path.join(self.config["root_path"], self.config["val"]["dataset_path"])
        self.config["test"]["dataset_path"] = os.path.join(self.config["root_path"], self.config["test"]["dataset_path"])
        self.config['val']['osm_path'] = os.path.join(self.config["root_path"], self.config['val']['osm_path'])
        self.config['worldfloods_folder'] = os.path.join(self.config["root_path"], self.config['worldfloods_folder'])

        # Load the model
        if self.config["train"]["model_ckpt"] is not None:
            self.config["train"]["model_ckpt"] = os.path.join(self.config["root_path"], self.config["train"]["model_ckpt"])
            self.model.load_state_dict(torch.load(self.config["train"]["model_ckpt"]), strict=False)
            logger.info(f'Model loaded from checkpoint: {self.config["train"]["model_ckpt"]}')
        
        if self.config["test"]["model_ckpt"] is not None:
            self.config["test"]["model_ckpt"] = os.path.join(self.config["root_path"], self.config["test"]["model_ckpt"])

    def pipeline(self) -> None:
        # Create new folder and init wandb
        self.config["val"]["output_dir"] = dir_checker(self.config["val"]["output_dir"], self.config_path)
        if self.wandb_log:
            wandb.login(key=self.wandb_token)
            wandb.init(
                entity="knife_team",
                project="skoltech_floods",
                config=self.config
            )

        self.t_loader = get_dataloader(config=self.config, data_split="train", batch_size=self.config["train_params"]["batch_size"])
        self.v_loader = get_dataloader(config=self.config, data_split="val", batch_size=self.config["val"]["batch_size"])

        self.state = "running"

        self.pbar = trange(
            self.config["train"]["epochs"], disable=(not self.config["progress_bar"]), position=0, leave=True
        )
        for epoch in self.pbar:
            if self.state in ["early_stopped", "interrupted", "finished"]:
                return

            self.postfix["Epoch"] = epoch
            self.pbar.set_postfix(self.postfix)

            try:
                self.train_procedure()
            except KeyboardInterrupt:
                logger.warning("\nKeyboard Interrupt detected. Attempting gracefull shutdown...")
                self.state = "interrupted"
            except Exception as err:
                raise (err)

            if self.state == "interrupted":
                self.validation_procedure()
                self.pbar.set_postfix(
                    {k: self.postfix[k] for k in self.postfix.keys() & {"train_loss_step", "total_f1", "f1_water", "avg_f1_business", "pre_f1", "post_f1", "f1", "iou"}}
                )

        self.state = "finished"

    def test(self) -> None:
        self.test_loader = get_dataloader(config=self.config, data_split="test", batch_size=self.config["test"]["batch_size"])
        self.test_results: TestResults = {}

        if self.best_model_path is not None:
            self.model.load_state_dict(torch.load(self.best_model_path), strict=False)
            logger.info(f"Best model loaded from checkpoint: {self.best_model_path}")
        elif self.config["test"]["model_ckpt"] is not None:
            self.model.load_state_dict(torch.load(self.config["test"]["model_ckpt"]), strict=False)
            logger.info(f'Model loaded from checkpoint: {self.config["test"]["model_ckpt"]}')
        elif self.state == "initializing":
            logger.warning("Warning: Testing with random weights")

        self.state = "running"
        self.test_procedure()
        self.state = "finished"

    def train_procedure(self) -> None:
        self.model.train()
        train_loss_list = []
        for step, batch in tqdm(
            enumerate(self.t_loader),
            total=len(self.t_loader),
            disable=(not self.config["progress_bar"]),
            position=2,
            leave=False,
        ):
            train_step = self.training_step(batch)
            self.postfix["train_loss_step"] = float(f"{train_step:.3f}")
            train_loss_list.append(train_step)
            self.pbar.set_postfix(
                {k: self.postfix[k] for k in self.postfix.keys() & {"train_loss_step", "total_f1", "f1_water", "avg_f1_business", "pre_f1", "post_f1", "f1", "iou"}}
            )
            if step % self.config["train"]["log_steps"] == 0:
                save_logs(
                    dict(
                        epoch=self.postfix["Epoch"],
                        step=step,
                        train_loss_step=f"{train_step:.3f}",
                    ),
                    output_dir=self.config["val"]["output_dir"],
                    name="log_steps",
                )
                if self.wandb_log:
                    wandb.log({"train_loss": train_step})
        train_loss = torch.tensor(train_loss_list)
        self.postfix["train_loss"] = train_loss.mean().item()
        self.validation_procedure()
        self.overfit_check()
        self.pbar.set_postfix({k: self.postfix[k] for k in self.postfix.keys() & {"train_loss_step", "total_f1", "f1_water", "avg_f1_business", "pre_f1", "post_f1", "f1", "iou"}})

    def training_step(self, batch: BatchDict) -> Dict[str, float]:
        preds = self.model.forward(batch["image"].to(self.config["device"]))
        loss = self.loss(preds, batch["mask"].to(self.config["device"]))

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item()

    def validation_procedure(self) -> None:
        self.model.eval()
        all_preds = []
        all_labels = []
        all_coords = []
        all_indices = []
        for batch in tqdm(self.v_loader, disable=(not self.config["progress_bar"]), position=1, leave=False):
            preds = self.validation_step(batch)
            preds = (preds > 0.5).astype(float)
            all_preds.append(preds)
            all_labels.append(batch["mask"].numpy().squeeze())
            all_coords.append(np.array(list(zip(batch["coords"][0], batch["coords"][1]))))
            all_indices.append(batch["image_index"].numpy())
        metrics, masks = calculate_metrics(all_labels, all_preds, all_coords, all_indices,
                                    self.config['train_params']['tile_size'], self.config['val']['osm_path'])

        for key, value in metrics.items():
            self.postfix[key] = value

        logger.info(
            f"\n{' Validation Results ':=^50}\n"
            + "\n".join([f'"{key}": {value}' for key, value in self.postfix.items()])
            + f"\n{' End of Validation ':=^50}\n"
        )
        if self.wandb_log:
            wandb.log({f"val_{key}": value for key, value in metrics.items()})

        if self.config["val"]["save_val_outputs"]:
            save_predictions(masks, output_dir=self.config["val"]["output_dir"])
            save_logs(self.postfix, output_dir=self.config["val"]["output_dir"])
            visualize_model_predictions(self.model, self.config)
        self.model.train()

    def validation_step(self, batch):
        preds = self.model.forward(batch["image"].to(self.config["device"]))
        return preds.detach().cpu().numpy().squeeze()

    def test_procedure(self) -> None:
        self.model.eval()
        all_preds = []
        all_coords = []
        all_indices = []
        filenames_set = set()
        all_filenames = []
        for batch in tqdm(self.test_loader, disable=(not self.config["progress_bar"]), position=1, leave=False):
            preds = self.validation_step(batch)
            preds = (preds > 0.5).astype(float)
            all_preds.append(preds)
            all_coords.append(np.array(list(zip(batch["coords"][0], batch["coords"][1]))))
            all_indices.append(batch["image_index"].numpy())
            for filename in batch["test_file_path"]:
                if filename not in filenames_set:
                    all_filenames.append(filename)
                    filenames_set.add(filename)
        masks = make_full_masks(None, all_preds, all_coords, all_indices,
                                self.config['train_params']['tile_size'])

        logger.info(f"Test with model {self.config['test']['model_name']} completed!")

        save_test_predictions(masks, all_filenames, dataset_path=self.config["test"]["dataset_path"],
                                                    model_name="preds_" + self.config["test"]["model_name"])
        #visualize_model_predictions(self.model, self.config)
        self.model.train()
    
    def overfit_check(self) -> None:
        validation_metric_name = 'f1_water'
        if self.early_stop(self.postfix[validation_metric_name]):
            logger.info(f"\nValidation not improved for {self.early_stop.patience} consecutive epochs. Stopping...")
            self.state = "early_stopped"

        if self.early_stop.counter > 0:
            logger.info(f"\nValidation metric ({validation_metric_name}) was not improved")
        else:
            logger.info(f"\nMetric improved. New best score: {self.early_stop.max_validation_metric:.3f}")
            save_best_log(self.postfix, output_dir=self.config["val"]["output_dir"])

            logger.info("Saving model...")
            epoch = self.postfix["Epoch"]
            prev_model = deepcopy(self.best_model_path)
            self.best_model_path = os.path.join(
                self.config["val"]["output_dir"], "model", f"best-model-epoch={epoch}.pt"
            )
            os.makedirs(os.path.dirname(self.best_model_path), exist_ok=True)
            torch.save(deepcopy(self.model.state_dict()), self.best_model_path)
            if prev_model is not None:
                os.remove(prev_model)

    def configure_optimizers(self) -> torch.optim.Optimizer:
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.config["train_params"]["learning_rate"])

        return optimizer
