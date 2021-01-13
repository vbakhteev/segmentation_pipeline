import torch
from torch import nn
from effdet.efficientdet import BiFpn, get_feature_info, HeadNet
from effdet.config import get_efficientdet_config

from .base_decoder import BaseDecoder, EncoderDecoder
from models.utils import change_layers_dim, get_layers_by_dim


bifpn_model_param_dict = dict(
    bifpn_d0=dict(
        name="bifpn_d0",
        fpn_channels=64,
        fpn_cell_repeats=3,
        box_class_repeats=3,
        pad_type="",
        redundant_bias=False,
    ),
)


class BiFPN(EncoderDecoder):
    """
    Only supports encoders from timm.
    """

    def __init__(
        self,
        encoder_name: str,
        n_dim: int,
        in_channels: int,
    ):
        super().__init__(encoder_name, n_dim, in_channels)

        config = get_bifpn_config()
        feature_info = get_feature_info(self.encoder.model)
        self.decoder = BiFPNDecoder(
            out_channels=88,
            config=config,
            feature_info=feature_info,
            n_dim=n_dim,
        )

    def forward(self, x):
        encoder_features = self.encoder(x)
        decoder_output = self.decoder([f for f in encoder_features])

        return encoder_features, decoder_output


class BiFPNDecoder(BaseDecoder):
    def __init__(self, out_channels, config, feature_info, n_dim=2):
        super().__init__(out_channels=out_channels)
        if n_dim == 2:
            self.bifpn, self.head = get_bifpn_2d(
                config, feature_info, out_channels // 9
            )
        elif n_dim == 3:
            self.bifpn = get_bifpn_3d(config, feature_info)

        self.head = BiFPNHead(out_channels, n_dim)

    def forward(self, features):
        fused_features = self.bifpn(features)
        out = self.head(fused_features)
        return out


class BiFPNHead(nn.Module):
    def __init__(self, out_channels, n_dim):
        super().__init__()
        self.upsampling = nn.ModuleList(
            [
                nn.Upsample(scale_factor=2 ** i, mode="bilinear", align_corners=False)
                for i in range(5)
            ]
        )
        Conv = get_layers_by_dim(n_dim)["conv"]
        self.conv_head = Conv(
            out_channels * 5,
            out_channels,
            kernel_size=1,
            bias=True,
        )
        self.activation = nn.SiLU()

    def forward(self, features):
        out = [self.upsampling[i](f) for i, f in enumerate(features)]
        out = torch.cat(out, dim=1)
        out = self.activation(self.conv_head(out))
        return out


def get_bifpn_2d(config, feature_info, out_channels):
    return BiFpn(config, feature_info), HeadNet(config, num_outputs=out_channels)


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
def get_bifpn_3d(config, feature_info, out_channels):
    return get_bifpn_2d(config, feature_info, out_channels)


def get_bifpn_config():
    h = get_efficientdet_config()
    return h
