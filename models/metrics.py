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


class BaseSegmentationMetric(pl.metrics.Metric):
    def __init__(
        self,
        metric_fn: Callable,
        mode: str,
        compute_on_step=True,
        dist_sync_on_step=False,
        threshold=0.5,
    ):
        super().__init__(
            compute_on_step=compute_on_step, dist_sync_on_step=dist_sync_on_step
        )

        assert mode in ("binary", "multiclass")

        self.metric_fn = metric_fn
        self.mode = mode
        self.threshold = threshold

        self.add_state("scores_sum", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0.0), dist_reduce_fx="sum")

    def update(self, preds: torch.Tensor, target: torch.Tensor) -> None:
        preds, target = self._input_format(preds, target)

        scores = self.metric_fn(preds, target)
        self.scores_sum += torch.sum(scores)
        self.total += scores.shape[0] if len(scores.shape) else 1

    def compute(self):
        return self.scores_sum / self.total

    def _input_format(self, preds, target):
        if self.mode == "binary":
            preds = preds.softmax(dim=1)
            preds = (preds[:, 1] > self.threshold).unsqueeze(1)

        elif self.mode == "multiclass":
            preds = preds.argmax(dim=1)

        preds = ohe_tensor(preds.long())
        target = ohe_tensor(target.long())

        return preds, target


def ohe_tensor(tensor):
    size = tensor.shape
    values = tensor.reshape(-1)
    n_values = torch.max(values) + 1
    ohe = torch.eye(n_values)[values]
    ohe = ohe.reshape(*size, n_values)

    return ohe


def intersection_over_union(outputs: torch.tensor, labels: torch.tensor):
    """Intersection over union metric
    Args:
        outputs (torch.tensor): Outputs of model. (N, C, H, W) or (N, C, H, W, Z)
        labels (torch.tensor): Ground truth.      (N, C, H, W) or (N, C, H, W, Z)
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
