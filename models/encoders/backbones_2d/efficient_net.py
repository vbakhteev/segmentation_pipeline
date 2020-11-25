from efficientnet_pytorch import EfficientNet
from torch import nn

from models.encoders.base_encoder import BaseEncoder

efficient_net_stages = {
    "efficientnet-b0": (3, 5, 9, 16),
    "efficientnet-b1": (5, 8, 16, 23),
    "efficientnet-b2": (5, 8, 16, 23),
    "efficientnet-b3": (5, 8, 18, 26),
    "efficientnet-b4": (6, 10, 22, 32),
    "efficientnet-b5": (8, 13, 27, 39),
    "efficientnet-b6": (9, 15, 31, 45),
    "efficientnet-b7": (11, 18, 38, 55),
}


class EfficientNetEncoder(BaseEncoder):
    def __init__(self, model, stage_ids):
        super(EfficientNetEncoder, self).__init__()
        self.init_block = nn.Sequential(
            model._conv_stem,
            model._bn0,
            model._swish,
        )
        self.layer1 = nn.Sequential(*model._blocks[: stage_ids[0]])
        self.layer2 = nn.Sequential(*model._blocks[stage_ids[0] : stage_ids[1]])
        self.layer3 = nn.Sequential(*model._blocks[stage_ids[1] : stage_ids[2]])

        last_layers = [model._conv_head, model._bn1, model._swish]
        self.layer4 = nn.Sequential(
            *(list(model._blocks[stage_ids[2] :]) + last_layers)
        )

        self.out_channels = self.get_out_channels_2d()


def get_efficientnet(model_name):
    base_model = EfficientNet.from_pretrained(model_name)
    model = EfficientNetEncoder(base_model, efficient_net_stages[model_name])
    return model
