from torch import nn

from models.encoders.base_encoder import BaseEncoder
from models.encoders import get_encoder


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
