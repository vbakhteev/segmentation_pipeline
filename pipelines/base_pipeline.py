import copy

import pytorch_lightning as pl
import torch
import numpy as np

from data import get_dataloader
from models import (
    get_optimizer,
    get_scheduler,
)


class BasePipeline(pl.LightningModule):
    """Parent class for pipelines with single model, dataset, optimizer, scheduler"""

    def __init__(self, experiment):
        super().__init__()

        self.cfg = None
        self.criterion = None
        self.update_config(experiment["cfg"])

        self.logging_names = []
        self.logged_metrics = dict()
        self.setup_logging()

    def setup_logging(self):
        raise NotImplementedError

    def update_config(self, cfg):
        self.cfg = copy.deepcopy(cfg)
        self.hparams = dict(cfg=self.cfg)

        self.update_criterion()

    def update_criterion(self):
        self.criterion = None
        raise NotImplementedError

    def forward(self, input_):
        return self.model(input_)

    def training_step(self, batch, batch_idx):
        raise NotImplementedError

    def training_epoch_end(self, outputs: list):
        loss = torch.stack([o["loss"] for o in outputs]).mean()
        self.log("train_loss", loss)
        self.logged_metrics["train_loss"] += [loss.item()]

    def validation_step(self, batch, batch_idx):
        raise NotImplementedError

    def validation_epoch_end(self, outputs: list):
        self.logged_metrics["valid_loss"][-1] = np.mean(
            self.logged_metrics["valid_loss"][-1]
        )

        for metric_name in self.logging_names:
            metric_o = getattr(self, metric_name)
            self.logged_metrics[metric_name] += [metric_o.compute().item()]

    def test_step(self, batch, batch_idx):
        raise NotImplementedError

    def test_epoch_end(self, outputs):
        raise NotImplementedError

    def configure_optimizers(self):
        optimizer = get_optimizer(self.model.parameters(), self.cfg.optimizer[0])
        scheduler = get_scheduler(optimizer, self.cfg.scheduler[0])

        result = {"optimizer": optimizer, "lr_scheduler": scheduler}
        if "ReduceLROnPlateau" in self.cfg.scheduler[0]["cls"]:
            result["monitor"] = "valid_loss"

        return result

    def optimizer_zero_grad(self, current_epoch, batch_idx, optimizer, opt_idx):
        # TODO get params from optimizer
        for param in self.model.parameters():
            param.grad = None

    ####################
    # DATA RELATED HOOKS
    ####################

    def get_dataloader(self, stage):
        return get_dataloader(self.cfg, stage=stage)

    def train_dataloader(self):
        return self.get_dataloader("train")

    def val_dataloader(self):
        return self.get_dataloader("valid")

    def test_dataloader(self):
        return self.get_dataloader("test")
