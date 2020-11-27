from pytorchcv.model_provider import get_model as ptcv_get_model

from models.encoders.base_encoder import BaseEncoder


class DLAEncoder(BaseEncoder):
    def __init__(self, model):
        super(DLAEncoder, self).__init__()
        self.init_block = model.features.init_block
        self.layer1 = model.features.stage1
        self.layer2 = model.features.stage2
        self.layer3 = model.features.stage3
        self.layer4 = model.features.stage4

        self.out_channels = self.get_out_channels_2d()


def get_dla(model_name):
    base_model = ptcv_get_model(model_name, pretrained=True)
    model = DLAEncoder(base_model)
    return model
