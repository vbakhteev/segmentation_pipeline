import segmentation_models_pytorch as smp

from utils import dict_remove_key, object_from_dict
from .base_models import MultiHeadSegmentator
from .criterions import get_criterion
from .decoders import SegNet, HRNet, BiFPN, EncoderDecoderSMP
from .encoders import get_encoder
from .metrics import BaseSegmentationMetric, intersection_over_union, dice_score
from .modules import get_classification_head, get_segmentation_head
from .utils import change_layers_dim, load_state_dict

__all__ = [
    "get_optimizer",
    "get_scheduler",
    "get_metrics",
    "get_metric",
    "get_criterion",
]

smp_models = {
    "Unet": smp.Unet,
    "UnetPlusPlus": smp.UnetPlusPlus,
    "Linknet": smp.Linknet,
    "FPN": smp.FPN,
    "PSPNet": smp.PSPNet,
    "PAN": smp.PAN,
    "DeepLabV3": smp.DeepLabV3,
    "DeepLabV3Plus": smp.DeepLabV3Plus,
}

available_models_segmentation = {
    "SegNet": SegNet,
    "HRNet": HRNet,
    "BiFPN": BiFPN,
}

segmentation_metrics = {
    "intersection_over_union": intersection_over_union,
    "dice_score": dice_score,
}


def get_segmentation_model(model_cfg):
    name = model_cfg.name
    n_dim = model_cfg.n_dim
    params = model_cfg.params

    if name in smp_models:
        if n_dim == 2:
            model = smp_models[name](**params)
        elif n_dim == 3:
            model = get_smp_3d(name, **params)
        else:
            raise NotImplementedError(
                f"Model {name} from smp is not supported for {n_dim}D"
            )

        model = EncoderDecoderSMP(
            model, checkpointing=model_cfg.activations_checkpointing
        )

    elif name in available_models_segmentation:
        model = available_models_segmentation[name](n_dim=n_dim, **params)
        model.checkpointing = model_cfg.activations_checkpointing

    else:
        raise KeyError(f"Segmentation model {name} is not supported for {n_dim}D")

    if model_cfg.encoder_weights != "":
        load_state_dict(model.encoder, model_cfg.encoder_weights)

    model = MultiHeadSegmentator(
        model=model,
        n_dim=n_dim,
        seg_heads_cfgs=model_cfg.segmentation.heads,
        clf_heads_cfgs=model_cfg.classification.heads,
    )

    if model_cfg.model_weights != "":
        load_state_dict(model, model_cfg.model_weights, soft=True)

    return model


@change_layers_dim(
    from_dim=2,
    to_dim=3,
    layer_names=(
        "Conv",
        "BatchNorm",
        "MaxPool",
        "AvgPool",
        "ConvTranspose",
        "AdaptiveMaxPool",
        "AdaptiveAvgPool",
    ),
)
def get_smp_3d(name, **params):
    params["encoder_weights"] = None
    model = smp_models[name](**params)
    return model


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
