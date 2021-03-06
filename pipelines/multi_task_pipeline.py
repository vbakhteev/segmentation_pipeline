from typing import Tuple

from models import get_criterion, get_metrics
from .base_pipeline import BasePipeline


class MultiTaskPipeline(BasePipeline):
    @staticmethod
    def pipeline_tasks() -> Tuple[str]:
        """Returns names of tasks for specific pipeline. These tasks are matched to cfg.model.[task]
        For example: Segmentation pipeline implements both classification and segmentation, so
        result of this function would be: (`classification`, `segmentation`)
        """
        raise NotImplementedError

    def setup_logging(self):
        self.logging_names = []

        for task in self.pipeline_tasks():
            task_cfg = self.cfg.model[task]
            for head_cfg in task_cfg.heads:
                if "metrics" not in head_cfg:
                    continue

                logging_metrics = get_metrics(
                    head_cfg.metrics, prefix=head_cfg.target + "@"
                )
                self.logging_names += list(logging_metrics.keys())
                for name, metric in logging_metrics.items():
                    setattr(self, name, metric)

        self.logged_metrics = {
            k: [] for k in ["train_loss", "valid_loss"] + self.logging_names
        }

    def update_criterion(self) -> None:
        """Setup (dataset, target) -> criterion mapping
        """
        self.criterion = {}
        for task in self.pipeline_tasks():
            task_cfg = self.cfg.model.get(task)
            default_criterion_cfg = task_cfg.get("default_criterion", None)

            for head_cfg in task_cfg.heads:
                criterion_cfg = head_cfg.get("criterion", None) or default_criterion_cfg

                for dataset_id in head_cfg.datasets_ids:
                    self.criterion[(dataset_id, head_cfg.target)] = get_criterion(
                        criterion_cfg
                    )

    def one_pass(self, batch):
        image = batch["image"]
        model_outputs = self.model(image)

        dataset_id = batch["dataset_id"]
        losses, targets = dict(), dict()

        for target_name in batch.keys():
            if target_name not in model_outputs:
                continue

            preds = model_outputs[target_name]
            target = batch[target_name]

            criterion_key = (dataset_id, target_name)
            criterion = self.criterion[criterion_key]

            if target.size(1) == 1:
                target = target.squeeze(1)

            loss = criterion(preds, target)

            losses[criterion_key] = loss
            targets[target_name] = target

        # TODO add weights for losses
        total_loss = sum(loss for key, loss in losses.items())

        return total_loss, losses, model_outputs, targets

    def training_step(self, batch, batch_idx):
        loss, losses, _, _ = self.one_pass(batch)
        return loss

    def validation_step(self, batch, batch_idx):
        loss, losses, model_outputs, targets = self.one_pass(batch)
        self.log("valid_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        if batch_idx == 0:
            self.logged_metrics["valid_loss"].append([])
            self.last_model_outputs = model_outputs
        else:
            self.logged_metrics["valid_loss"][-1] += [loss.item()]

        for metric_name in self.logging_names:
            target_name = metric_name.split("@")[0]
            if target_name not in targets:
                continue

            pred = model_outputs[target_name]
            target = targets[target_name]
            metric_o = getattr(self, metric_name)

            score = metric_o(pred, target)
            self.log(metric_name, score, on_step=False, on_epoch=True, prog_bar=True)
