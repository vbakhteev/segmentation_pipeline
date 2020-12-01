import pydoc
from typing import Optional

from torch import nn

from .utils import get_layers_by_dim


def get_segmentation_head(
    head: str, in_channels: int, classes: int, n_dim: int, **params
):
    if head not in segmentation_heads:
        assert NotImplementedError(
            f"Segmentation head {head} is not supported. "
            f"Try: {list(segmentation_heads.keys())}"
        )

    seg_head = segmentation_heads[head](
        in_channels=in_channels,
        classes=classes,
        n_dim=n_dim,
        **params,
    )
    return seg_head


def get_classification_head(
    head: str, in_channels: int, classes: int, n_dim: int, **params
):
    if head not in classification_heads:
        assert NotImplementedError(
            f"Classification head {head} is not supported. "
            f"Try: {list(classification_heads.keys())}"
        )

    clf_head = classification_heads[head](
        in_channels=in_channels,
        classes=classes,
        n_dim=n_dim,
        **params,
    )
    return clf_head


def get_adaptive_pooling(n_dim: int, pooling_name: str = "avg"):
    if pooling_name not in ("max", "avg"):
        raise ValueError(
            f"Pooling should be one of ('max', 'avg'), got {pooling_name}."
        )

    pool_name = "adaptive_max_pool" if pooling_name == "max" else "adaptive_avg_pool"
    pool = get_layers_by_dim(n_dim)[pool_name](1)
    return pool


def get_activation(cls_name: str = "torch.nn.ReLU", **params):
    obj = pydoc.locate(cls_name)(**params)  # type: ignore
    return obj


class LinearClassificationHead(nn.Sequential):
    def __init__(
        self,
        in_channels: int,
        classes: int,
        n_dim: int,
        pooling: str = "avg",
        dropout: float = 0.2,
    ):
        if pooling not in ("max", "avg"):
            raise ValueError(
                "Pooling should be one of ('max', 'avg'), got {}.".format(pooling)
            )

        pool = get_adaptive_pooling(n_dim=n_dim, pooling_name=pooling)
        dropout = nn.Dropout(p=dropout, inplace=True) if dropout else nn.Identity()
        linear = nn.Linear(in_channels, classes, bias=True)

        super().__init__(
            pool,
            nn.Flatten(),
            dropout,
            linear,
        )


class MultiLayerClassificationHead(nn.Sequential):
    def __init__(
        self,
        in_channels: int,
        classes: int,
        n_dim: int,
        layer_sizes: list,
        pooling: str = "avg",
        dropout: float = 0.2,
        activation: str = "torch.nn.ReLU",
        activation_params: Optional[dict] = None,
    ):
        if activation_params is None:
            activation_params = dict()
        pool = get_adaptive_pooling(n_dim=n_dim, pooling_name=pooling)
        dropout = nn.Dropout(p=dropout, inplace=True) if dropout else nn.Identity()

        prev_size = in_channels
        layers = [
            pool,
            nn.Flatten(),
            dropout,
        ]
        for size in layer_sizes:
            layers += [
                nn.Linear(prev_size, size),
                get_activation(activation, **activation_params),
            ]
            prev_size = size

        layers += [nn.Linear(prev_size, classes)]

        super().__init__(*layers)


class ConvUpsampleSegmentationHead(nn.Sequential):
    def __init__(
        self,
        in_channels: int,
        classes: int,
        n_dim: int,
        kernel_size: int = 3,
        upsampling: int = 1,
    ):
        Conv = get_layers_by_dim(n_dim)["conv"]
        conv = Conv(
            in_channels, classes, kernel_size=kernel_size, padding=kernel_size // 2
        )
        upsampling = (
            nn.Upsample(scale_factor=upsampling, mode="bilinear")
            if upsampling > 1
            else nn.Identity()
        )
        super().__init__(conv, upsampling)


segmentation_heads = {
    "conv_upsample": ConvUpsampleSegmentationHead,
}

classification_heads = {
    "linear": LinearClassificationHead,
    "multi_layer": MultiLayerClassificationHead,
}
