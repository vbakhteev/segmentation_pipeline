import pretrainedmodels
from torch import nn

from .base_ import BaseEncoder


class ResNextEncoder(BaseEncoder):
    def __init__(self, model):
        super(ResNextEncoder, self).__init__()
        self.init_block = nn.Sequential(
            model[0],
            model[1],
            model[2],
            model[3],
        )
        self.layer1 = model[4]
        self.layer2 = model[5]
        self.layer3 = model[6]
        self.layer4 = model[7]


def get_resnext(model_name):
    base_model = pretrainedmodels.__dict__[model_name]()
    model = ResNextEncoder(base_model.features)
    return model
