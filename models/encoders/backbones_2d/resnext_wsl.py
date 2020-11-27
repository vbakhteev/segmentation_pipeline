import torch
from torch import nn

from models.encoders.base_encoder import BaseEncoder


class ResNextEncoder(BaseEncoder):
    def __init__(self, model):
        super(ResNextEncoder, self).__init__()
        self.init_block = nn.Sequential(
            model.conv1,
            model.bn1,
            model.relu,
            model.maxpool,
        )
        self.layer1 = model.layer1
        self.layer2 = model.layer2
        self.layer3 = model.layer3
        self.layer4 = model.layer4

        self.out_channels = self.get_out_channels_2d()


def get_resnext_wsl(model_name):
    base_model = torch.hub.load("facebookresearch/WSL-Images", model_name)
    model = ResNextEncoder(base_model)
    return model
