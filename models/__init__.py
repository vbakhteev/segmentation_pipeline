import segmentation_models_pytorch as smp

from utils import dict_remove_key, object_from_dict
from .base_models import MultiHeadSegmentator
from .criterions import get_criterion
from .decoders import SegNet, EncoderDecoderSMP
from .encoders import get_encoder
from .metrics import BaseSegmentationMetric, intersection_over_union
from .modules import get_classification_head, get_segmentation_head

__all__ = [
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


def get_segmentation_model(model_cfg):
    name = model_cfg.name
    n_dim = model_cfg.n_dim
    params = model_cfg.params

    if n_dim == 2 and name in available_2d_models_segmentation:
        model = available_2d_models_segmentation[name](**params)
        model = EncoderDecoderSMP(model)

    elif name in available_models_segmentation:
        model = available_models_segmentation[name](n_dim=n_dim, **params)

    else:
        raise KeyError(f"Segmentation model {name} is not supported for {n_dim}D")

    return MultiHeadSegmentator(
        model=model,
        n_dim=n_dim,
        seg_heads_cfgs=model_cfg.segmentation.heads,
        clf_heads_cfgs=model_cfg.classification.heads,
    )


def get_optimizer(params, cfg_optimizer):
    return object_from_dict(cfg_optimizer, params=params)


def get_scheduler(optimizer, cfg_scheduler):
    return object_from_dict(cfg_scheduler, optimizer=optimizer)


def get_metrics(cfg_metrics: list, prefix="") -> dict:
    metrics_ = dict()
    for cfg_metric in cfg_metrics:
        metrics_[prefix + cfg_metric.name] = get_metric(cfg_metric)

    return metrics_


def get_metric(cfg_metric):
    name = cfg_metric.name

    if name in segmentation_metrics:
        metric_fn = segmentation_metrics[name]
        params = dict_remove_key(cfg_metric, "name")
        return BaseSegmentationMetric(metric_fn=metric_fn, **params)

    else:
        raise KeyError(f"Metric {name} is not supported")
