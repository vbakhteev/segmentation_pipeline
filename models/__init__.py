import segmentation_models_pytorch as smp
from torch import nn

from utils import dict_remove_key, object_from_dict
from .criterions import get_criterion
from .decoders import SegNet
from .encoders import get_encoder
from .metrics import BaseSegmentationMetric, intersection_over_union

__all__ = [
    "get_model",
    "get_optimizer",
    "get_scheduler",
    "get_metrics",
    "get_metric",
    "get_criterion",
]

available_2d_models_segmentation = {
    "Unet": smp.Unet,
    # "UnetPlusPlus": smp.UnetPlusPlus,
    "Linknet": smp.Linknet,
    "FPN": smp.FPN,
    "PSPNet": smp.PSPNet,
    "PAN": smp.PAN,
    "DeepLabV3": smp.DeepLabV3,
    "DeepLabV3Plus": smp.DeepLabV3Plus,
}

available_models_segmentation = {"SegNet": SegNet}

segmentation_metrics = {"intersection_over_union": intersection_over_union}


def get_model(name: str, model_type: str, params: dict, n_dim: int) -> nn.Module:
    assert model_type in ("segmentation",)

    if model_type == "segmentation":
        return get_segmentation_model(name=name, params=params, n_dim=n_dim)

    else:
        raise KeyError(f"Model type {model_type} is not supported")


def get_segmentation_model(name: str, params: dict, n_dim: int):
    if n_dim == 2 and name in available_2d_models_segmentation:
        model = available_2d_models_segmentation[name](**params)

    elif name in available_models_segmentation:
        model = available_models_segmentation[name](n_dim=n_dim, **params)

    else:
        raise KeyError(f"Segmentation model {name} is not supported for {n_dim}D")

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

    if name in segmentation_metrics:
        metric_fn = segmentation_metrics[name]
        params = dict_remove_key(cfg_metric, "name")
        return BaseSegmentationMetric(metric_fn=metric_fn, **params)

    else:
        raise KeyError(f"Metric {name} is not supported")
