import pretrainedmodels
from torch import nn

from models.encoders.base_encoder import BaseEncoder


class ResNetEncoder(BaseEncoder):
    def __init__(self, model):
        super(ResNetEncoder, self).__init__()
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


def get_out_channels_resnet_block(block):
    try:
        return block.conv3.out_channels
    except AttributeError:
        return block.conv2.out_channels


def get_resnet(model_name):
    base_model = pretrainedmodels.__dict__[model_name]()
    model = ResNetEncoder(base_model)
    return model
