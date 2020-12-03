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
        self.decoder.out_channels = self.get_decoder_out_channels()

    def get_decoder_out_channels(self):
        for module in self.encoder.modules():
            if isinstance(module, nn.Conv2d):
                break
        in_channels = module.in_channels

        mock_tensor = torch.zeros((1, in_channels, 64, 64))
        _, decoder_output = self.forward(mock_tensor)
        return decoder_output.shape[1]

    def forward(self, x):
        encoder_features = self.encoder(x)
        decoder_output = self.decoder(*encoder_features)

        return encoder_features, decoder_output
