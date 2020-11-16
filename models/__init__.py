import segmentation_models_pytorch as smp
from torch import nn

from utils import dict_remove_key, object_from_dict
from .criterions import get_criterion
from .metrics import BaseMetric, intersection_over_union
from .metrics import intersection_over_union, BaseMetric

__all__ = [
    "get_model",
    "get_optimizer",
    "get_scheduler",
    "get_metrics",
    "get_metric",
    "get_criterion",
]

available_metrics = {"intersection_over_union": intersection_over_union}

available_2d_models = {
    "Unet": smp.Unet,
    "Linknet": smp.Linknet,
    "FPN": smp.FPN,
    "PSPNet": smp.PSPNet,
    "PAN": smp.PAN,
    "DeepLabV3": smp.DeepLabV3,
    "DeepLabV3Plus": smp.DeepLabV3Plus,
}

available_3d_models = {}  # type: ignore


def get_model(cfg):
    if cfg.model.dim == 2:
        name2model = available_2d_models
    else:
        name2model = available_3d_models

    name = cfg.model.name
    if name not in name2model:
        raise KeyError(f"Model {name} is not supported for {cfg.model.dim}D")

    model_cls = name2model[name]
    model = model_cls(**cfg.model.params)

    return BaseModel(model)


class BaseModel(nn.Module):
    def __init__(self, model):
        super(BaseModel, self).__init__()
        self.model = model

    def forward(self, batch):
        return self.model(batch["image"])


def get_optimizer(params, cfg_optimizer):
    return object_from_dict(cfg_optimizer, params=params)


def get_scheduler(optimizer, cfg_scheduler):
    return object_from_dict(cfg_scheduler, optimizer=optimizer)


def get_metrics(cfg_metrics: list) -> dict:
    metrics_ = dict()
    for cfg_metric in cfg_metrics:
        metrics_[cfg_metric.name] = get_metric(cfg_metric)

    return metrics_


def get_metric(cfg_metric):
    name = cfg_metric.name
    if name not in available_metrics:
        raise KeyError(f"Metric {name} is not supported")

    metric_fn = available_metrics[name]
    params = dict_remove_key(cfg_metric, "name")
    return BaseMetric(metric_fn, **params)
