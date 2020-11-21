import copy

import pytorch_lightning as pl
import torch

from data import get_dataloader
from models import (
    get_criterion,
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
            model_type="segmentation",
            name=self.cfg.model.name,
            n_dim=self.cfg.model.n_dim,
            params=self.cfg.model.params,
        )

        logging_metrics = get_metrics(self.cfg.logging.metrics)
        self.logging_names = list(logging_metrics.keys())
        for name, metric in logging_metrics.items():
            setattr(self, name, metric)
        self.logged_metrics = {
            k: [] for k in ["train_loss"] + list(logging_metrics.keys())
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
        self.log("valid_loss", loss, on_step=False, on_epoch=True, prog_bar=True)

        for metric_name in self.logging_names:
            metric_o = getattr(self, metric_name)
            score = metric_o(logits_mask, mask)
            self.log(metric_name, score, on_step=False, on_epoch=True, prog_bar=True)

    def validation_epoch_end(self, outputs: list):
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
        # Faster than optimizer.zero_grad()
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
