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
    def __init__(
        self,
        metric_fn: Callable,
        compute_on_step=True,
        dist_sync_on_step=False,
        threshold=0.5,
    ):
        super(BaseMetric, self).__init__(
            compute_on_step=compute_on_step, dist_sync_on_step=dist_sync_on_step
        )

        self.metric_fn = metric_fn
        self.threshold = threshold

        self.add_state("scores_sum", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0.0), dist_reduce_fx="sum")

    def update(self, preds: torch.Tensor, target: torch.Tensor) -> None:
        # preds, target = self._input_format(preds, target)

        if self.threshold is not None:
            preds = (preds > self.threshold).int()
        target = target.int()

        scores = self.metric_fn(preds, target)

        self.scores_sum += torch.sum(scores)
        self.total += scores.shape[0] if len(scores.shape) else 1

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
    # Binary segmentation case
    if len(outputs.shape) - len(labels.shape) == 1:
        labels = labels.unsqueeze(1)

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
