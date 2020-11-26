from torch import nn

from .utils import get_layers_by_dim


class SegmentationHead(nn.Sequential):
    def __init__(self, in_channels, out_channels, n_dim, kernel_size=3, upsampling=1):
        Conv = get_layers_by_dim(n_dim)["conv"]
        conv2d = Conv(
            in_channels, out_channels, kernel_size=kernel_size, padding=kernel_size // 2
        )
        upsampling = (
            nn.Upsample(scale_factor=upsampling, mode="bilinear")
            if upsampling > 1
            else nn.Identity()
        )
        super().__init__(conv2d, upsampling)


class ClassificationHead(nn.Sequential):
    def __init__(self, in_channels, classes, n_dim, pooling="avg", dropout=0.2):
        if pooling not in ("max", "avg"):
            raise ValueError(
                "Pooling should be one of ('max', 'avg'), got {}.".format(pooling)
            )

        pool_name = "adaptive_max_pool" if pooling == "max" else "adaptive_avg_pool"
        pool = get_layers_by_dim(n_dim)[pool_name](1)
        dropout = nn.Dropout(p=dropout, inplace=True) if dropout else nn.Identity()
        linear = nn.Linear(in_channels, classes, bias=True)

        super().__init__(
            pool,
            nn.Flatten(),
            dropout,
            linear,
        )
