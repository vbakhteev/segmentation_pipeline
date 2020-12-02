import segmentation_models_pytorch.base as smp_base
import torch
from torch import nn

from models.encoders import get_encoder
from models.encoders.base_encoder import BaseEncoder


class BaseDecoder(nn.Module):
    def __init__(self, out_channels: int):
        super().__init__()
        self.out_channels = out_channels


class EncoderDecoder(nn.Module):
    def __init__(
        self,
        encoder_name: str,
        n_dim: int,
        in_channels: int,
    ):
        super().__init__()

        self.encoder: BaseEncoder = get_encoder(
            encoder_name=encoder_name, n_dim=n_dim, num_channels=in_channels
        )
        self.decoder: BaseDecoder = nn.Identity()

    def forward(self, x):
        encoder_features = self.encoder(x)
        decoder_output = self.decoder(*encoder_features)

        return encoder_features, decoder_output


class EncoderDecoderSMP(nn.Module):
    def __init__(self, model: smp_base.SegmentationModel):
        super().__init__()

        self.encoder = model.encoder
        self.decoder = model.decoder

        _, decoder_output = self.forward(torch.zeros((1, 3, 64, 64)))
        self.decoder.out_channels = decoder_output.shape[1]

    def forward(self, x):
        encoder_features = self.encoder(x)
        decoder_output = self.decoder(*encoder_features)

        return encoder_features, decoder_output
