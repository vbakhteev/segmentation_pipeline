"""
To implement your custom metric you need define and implement function

    def custom_metric(outputs: torch.tensor, labels: torch.tensor) -> torch.tensor

and this function should return vector of size (batch_size,) with computed score for each sample.
After that, you need to add implemented function into dict `metrics`.
"""

from typing import Callable

import pytorch_lightning as pl
import torch

EPS = 1e-8


class BaseMetric(pl.metrics.Metric):
    def __init__(self, metric_fn: Callable, dist_sync_on_step=False, threshold=0.5):
        super().__init__(dist_sync_on_step=dist_sync_on_step)

        self.metric_fn = metric_fn
        self.threshold = threshold

        self.add_state("scores_sum", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0.0), dist_reduce_fx="sum")

    def update(self, preds: torch.Tensor, target: torch.Tensor):
        preds, target = self._input_format(preds, target)

        if self.threshold is not None:
            preds = (preds > self.threshold).int()
        target = target.int()

        scores = self.metric_fn(preds, target)

        self.scores_sum += torch.sum(scores)
        self.total += target.shape[0]

    def compute(self):
        return self.scores_sum / self.total


def intersection_over_union(outputs: torch.tensor, labels: torch.tensor):
    """Intersection over union metric
    size can be (BATCH, H, W) or size (BATCH, Z, H, W)
    Args:
        outputs (torch.tensor): Outputs of model
        labels (torch.tensor): Ground truth
    returns:
         torch.tensor: vector of IoU for each sample
    """
    if outputs.shape != labels.shape:
        raise AttributeError(f"Shapes not equal: {outputs.shape} != {labels.shape}")
    outputs = outputs.int()
    labels = labels.int()

    bs = outputs.shape[0]
    outputs = outputs.reshape((bs, -1))
    labels = labels.reshape((bs, -1))
    intersection = (outputs & labels).sum(1).float()
    union = (outputs | labels).sum(1).float()

    iou = (intersection + EPS) / (union + EPS)
    return iou
