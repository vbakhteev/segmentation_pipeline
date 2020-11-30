from torch import nn

from models.base_models import EncoderDecoder
from models.utils import get_layers_by_dim
from .base_decoder import BaseDecoder


class SegNet(EncoderDecoder):
    def __init__(
        self,
        encoder_name: str,
        n_dim: int,
        in_channels: int,
    ):
        super().__init__(encoder_name, n_dim, in_channels)

        self.decoder = SegNetDecoder(
            in_channels=self.encoder.out_channels[-1],
            out_channels=32,
            n_dim=n_dim,
        )


def upsample_conv_bn_relu(in_channels, out_channels, n_dim=2):
    layers = get_layers_by_dim(n_dim=n_dim)
    Pad = layers["pad"]
    Conv = layers["conv"]
    BN = layers["batch_norm"]

    return nn.Sequential(
        nn.Upsample(scale_factor=2, mode="nearest"),
        Pad(1),
        Conv(in_channels, out_channels, kernel_size=3, stride=1, padding=0, bias=False),
        BN(out_channels),
        nn.ReLU(inplace=True),
    )


class SegNetDecoder(BaseDecoder):
    def __init__(self, in_channels, out_channels, n_dim=2):
        super().__init__(out_channels)

        layers = get_layers_by_dim(n_dim=n_dim)
        Conv = layers["conv"]
        BN = layers["batch_norm"]

        self.up_layer = nn.Sequential(
            Conv(
                in_channels,
                out_channels * 16,
                kernel_size=1,
                stride=1,
                padding=0,
                bias=True,
            ),
            upsample_conv_bn_relu(out_channels * 16, out_channels * 16, n_dim=n_dim),
            upsample_conv_bn_relu(out_channels * 16, out_channels * 8, n_dim=n_dim),
            upsample_conv_bn_relu(out_channels * 8, out_channels * 4, n_dim=n_dim),
            upsample_conv_bn_relu(out_channels * 4, out_channels * 2, n_dim=n_dim),
            upsample_conv_bn_relu(out_channels * 2, out_channels, n_dim=n_dim),
            Conv(
                out_channels,
                out_channels,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False,
            ),
            BN(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, *features):
        x = features[-1]
        x = self.up_layer(x)

        return x
