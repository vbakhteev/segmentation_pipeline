import segmentation_models_pytorch.base as smp_base
import torch
from torch import nn
from torch.utils.checkpoint import checkpoint

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
        checkpointing: bool = False,
    ):
        super().__init__()

        self.checkpointing = checkpointing
        self.encoder: BaseEncoder = get_encoder(
            encoder_name=encoder_name, n_dim=n_dim, num_channels=in_channels
        )
        self.decoder: BaseDecoder = nn.Identity()

    def get_encoder_features(self, x):
        return tuple(self.encoder(x))

    def forward(self, x):
        if self.checkpointing:
            encoder_features = checkpoint(self.get_encoder_features, x)
        else:
            encoder_features = self.encoder(x)

        decoder_output = self.decoder(*encoder_features)
        return encoder_features, decoder_output


class EncoderDecoderSMP(EncoderDecoder):
    def __init__(self, model: smp_base.SegmentationModel, checkpointing: bool = False):
        # Useless mock parameters to init nn.Module. Ignore them
        super().__init__("resnet18", 2, 1, checkpointing)
        self.encoder = model.encoder
        self.decoder = model.decoder
        self.decoder.out_channels = self.get_decoder_out_channels()

    def get_decoder_out_channels(self):
        n_dim = -1
        for module in self.encoder.modules():
            if isinstance(module, nn.Conv2d):
                n_dim = 2
                break
            elif isinstance(module, nn.Conv3d):
                n_dim = 3
                break

        in_channels = module.in_channels

        if n_dim == 2:
            mock_tensor = torch.zeros((1, in_channels, 128, 128))
        elif n_dim == 3:
            mock_tensor = torch.zeros((1, in_channels, 128, 128, 128))
        else:
            raise NotImplementedError(
                "Update method EncoderDecoderSMP.get_decoder_out_channels"
            )

        self.eval()
        with torch.no_grad():
            _, decoder_output = self.forward(mock_tensor)
        return decoder_output.shape[1]
