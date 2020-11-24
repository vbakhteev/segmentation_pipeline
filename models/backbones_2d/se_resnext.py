import pretrainedmodels

from .base_ import BaseEncoder


class SEResNextEncoder(BaseEncoder):
    def __init__(self, model):
        super(SEResNextEncoder, self).__init__()
        self.init_block = model.layer0
        self.layer1 = model.layer1
        self.layer2 = model.layer2
        self.layer3 = model.layer3
        self.layer4 = model.layer4


def get_seresnext(model_name):
    base_model = pretrainedmodels.__dict__[model_name]()
    model = SEResNextEncoder(base_model)
    return model
