import pretrainedmodels
from torch import nn

from models.encoders.base_encoder import BaseEncoder

# Hacker
nn.Conv2d = nn.Conv3d
nn.BatchNorm2d = nn.BatchNorm3d
nn.MaxPool2d = nn.MaxPool3d


class ResNetEncoder(BaseEncoder):
    def __init__(self, model):
        super(ResNetEncoder, self).__init__()
        self.init_block = nn.Sequential(
            model.conv1,
            model.bn1,
            model.relu,
        )
        self.layer1 = model.layer1
        self.layer2 = model.layer2
        self.layer3 = model.layer3
        self.layer4 = model.layer4

        self.out_channels = self.get_out_channels_3d()


def get_resnet(model_name):
    base_model = pretrainedmodels.__dict__[model_name](pretrained=None)
    model = ResNetEncoder(base_model)
    return model
