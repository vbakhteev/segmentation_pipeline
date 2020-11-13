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

        self.model = get_model(self.cfg)
        self.checkpoint_metric = get_metric(self.cfg.checkpointing.metric)
        self.logging_metrics = get_metrics(self.cfg.logging.metrics)

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
        loss = torch.cat(outputs).mean()
        self.log("train_loss", loss)

    def validation_step(self, batch, batch_idx):
        raise NotImplementedError
        mask = batch["mask"]
        logits_mask = self.model(batch)

        loss = self.criterion(logits_mask, mask)
        metric = self.checkpoint_metric(logits_mask, mask)

        for metric_name, metric_o in self.logging_metrics:
            metric_o(logits_mask, mask)
            self.log(metric_name, metric_o, on_step=True, on_epoch=True)

        results = {
            "val_loss": loss,
            "val_metric": metric,
            "progress_bar": {"val_metric": metric},
        }
        return results

    def validation_epoch_end(self, outputs: list):
        result = dict()
        for n in ("val_loss", "val_metric"):
            result[n] = torch.stack([x[n] for x in outputs]).mean()

        return result

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

    def prepare_data(self):
        # read csv or whatever with train/valid/test split
        raise NotImplementedError

        self.train_filenames = []
        self.valid_filenames = []
        self.test_filenames = []

    def train_dataloader(self):
        return get_dataloader(self.cfg, train=True, filenames=self.train_filenames)

    def val_dataloader(self):
        return get_dataloader(self.cfg, train=False, filenames=self.valid_filenames)

    def test_dataloader(self):
        return get_dataloader(self.cfg, train=False, filenames=self.test_filenames)
