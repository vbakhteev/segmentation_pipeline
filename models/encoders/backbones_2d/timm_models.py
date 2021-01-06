import timm

from models.encoders.base_encoder import BaseEncoder


class TimmEncoder(BaseEncoder):
    def __init__(self, model_name):
        super().__init__()
        self.model = timm.create_model(model_name, features_only=True, pretrained=True)
        self.out_channels = self.get_out_channels_2d()

    def forward(self, x, get_pyramid=True):
        x0, x1, x2, x3, x4 = self.model(x)

        if get_pyramid:
            return [x0, x1, x2, x3, x4]
        else:
            return x4


def get_timm_model(model_name):
    model = TimmEncoder(model_name)
    return model
