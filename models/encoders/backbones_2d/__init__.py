import torch
import pretrainedmodels
from torch import nn

from .resnet import get_resnet
from .resnext import get_resnext
from .resnext_wsl import get_resnext_wsl
from .se_resnext import get_seresnext
from .efficient_net import get_efficientnet


resnets = [f"resnet{i}" for i in (18, 34, 50, 101, 152)]

resnexts = ["resnext101_32x4d", "resnext101_64x4d"]

resnexts_wsl = [f"resnext101_32x{d}d_wsl" for d in (8, 16, 32, 48)]

se_resexts = ["se_resnext50_32x4d", "se_resnext101_32x4d"]

eff_nets = [f"efficientnet-b{i}" for i in range(8)]

pretrained = [
    "densenet121",
    "densenet169",
    "densenet201",
    "densenet161",
    "dpn68",
    "dpn68b",
    "dpn92",
    "dpn98",
    "dpn131",
    "dpn107",
    "se_resnet50",
    "se_resnet101",
    "se_resnet152",
    "nasnetamobile",
    "nasnetalarge",
    "fbresnet152",
    "bninception",
    "cafferesnet101",
    "pnasnet5large",
    "xception",
    "senet154",
]


def get_backbone_2d(model_name):
    if model_name in resnets:
        model = get_resnet(model_name)

    elif model_name in resnexts:
        model = get_resnext(model_name)

    elif model_name in resnexts_wsl:
        model = get_resnext_wsl(model_name)

    elif model_name in se_resexts:
        model = get_seresnext(model_name)

    elif model_name in eff_nets:
        model = get_efficientnet(model_name)

    elif model_name in pretrained:
        raise NotImplementedError("Not yet standartized")
        model = pretrainedmodels.__dict__[model_name]()

    else:
        raise NotImplementedError("Backbone {} is not supported".format(model_name))

    return model


if __name__ == "__main__":
    model = get_backbone_2d("efficientnet-b0")
    print(model)
    print(model(torch.zeros((1, 3, 128, 128)))[-1].shape)
