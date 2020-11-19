import copy

import pytorch_lightning as pl
import torch

from data import get_dataloader
from models import (
    get_criterion,
    get_metric,
    get_metrics,
    get_model,
    get_optimizer,
    get_scheduler,
)


class Segmentator(pl.LightningModule):
    def __init__(self, experiment):
        super(Segmentator, self).__init__()

        self.cfg = None
        self.criterion = None
        self.update_config(experiment["cfg"])
        self.prepared_data = None

        self.model = get_model(
            n_dim=self.cfg.model.dim,
            name=self.cfg.model.name,
            params=self.cfg.model.params,
        )
        self.checkpoint_metric = get_metric(self.cfg.checkpointing.metric)
        self.logging_metrics = get_metrics(self.cfg.logging.metrics)
        self.logged_metrics = {
            k: []
            for k in ["train_loss", "val_metric"] + list(self.logging_metrics.keys())
        }

    def update_config(self, cfg):
        self.cfg = copy.deepcopy(cfg)
        self.hparams = dict(cfg=self.cfg)

        self.criterion = get_criterion(self.cfg.criterion[0])

    def forward(self, input_):
        return self.model(input_)

    def training_step(self, batch, batch_idx):
        mask = batch["mask"]
        logits_mask = self.model(batch)
        loss = self.criterion(logits_mask, mask)

        return loss

    def training_epoch_end(self, outputs: list):
        loss = torch.stack([o["loss"] for o in outputs]).mean()
        self.log("train_loss", loss)
        self.logged_metrics["train_loss"] += [loss.item()]

    def validation_step(self, batch, batch_idx):
        mask = batch["mask"]
        logits_mask = self.model(batch)

        loss = self.criterion(logits_mask, mask)
        metric = self.checkpoint_metric(logits_mask, mask)
        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val_metric", metric, on_step=False, on_epoch=True, prog_bar=True)

        for metric_name, metric_o in self.logging_metrics.items():
            score = metric_o(logits_mask, mask)
            self.log(metric_name, score, on_step=False, on_epoch=True)

    def validation_epoch_end(self, outputs: list):
        self.logged_metrics["val_metric"] += [self.checkpoint_metric.compute().item()]

        for metric_name, metric_o in self.logging_metrics.items():
            self.logged_metrics[metric_name] += [metric_o.compute().item()]

    def test_step(self, batch, batch_idx):
        raise NotImplementedError

    def test_epoch_end(self, outputs):
        raise NotImplementedError

    def configure_optimizers(self):
        optimizer = get_optimizer(self.model.parameters(), self.cfg.optimizer[0])
        scheduler = get_scheduler(optimizer, self.cfg.scheduler[0])
        return {"optimizer": optimizer, "lr_scheduler": scheduler}

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
