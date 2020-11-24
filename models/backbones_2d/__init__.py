import torch
import pretrainedmodels
from torch import nn

from .dla import get_dla
from .resnet import get_resnet
from .resnext import get_resnext
from .resnext_wsl import get_resnext_wsl
from .se_resnext import get_seresnext
from .efficient_net import get_efficientnet


dla_nets = [f"dla{i}" for i in (34, 60, 102, 169)]

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


def get_backbone(model_name, num_channels=3):
    if model_name in resnets:
        model = get_resnet(model_name)

    elif model_name in resnexts:
        model = get_resnext(model_name)

    elif model_name in resnexts_wsl:
        model = get_resnext_wsl(model_name)

    elif model_name in se_resexts:
        model = get_seresnext(model_name)

    elif model_name in dla_nets:
        model = get_dla(model_name)

    elif model_name in eff_nets:
        model = get_efficientnet(model_name)

    elif model_name in pretrained:
        raise NotImplementedError("Not yet standartized")
        model = pretrainedmodels.__dict__[model_name]()

    else:
        raise NotImplementedError("Backbone {} is not supported".format(model_name))

    if num_channels == 3:
        pass

    elif num_channels == 1:
        named_conv = next(model.named_parameters())
        name = named_conv[0].split(".")[:-1]
        w = named_conv[1]  # (out_filters, 3, k, k)

        nn_module = model
        for n in name[:-1]:
            nn_module = getattr(nn_module, n)
        first_conv = next(nn_module.children())

        setattr(
            nn_module,
            name[-1],
            nn.Conv2d(
                1,
                w.shape[0],
                kernel_size=w.shape[2:],
                stride=first_conv.stride,
                padding=first_conv.padding,
                bias=first_conv.bias,
                dilation=first_conv.dilation,
                groups=first_conv.groups,
            ),
        )
        w = torch.sum(w, dim=1, keepdim=True)
        getattr(nn_module, name[-1]).weight = nn.Parameter(w)

    else:
        raise NotImplementedError(
            "Number of channels {} is not supported".format(num_channels)
        )

    return model


if __name__ == "__main__":
    model = get_backbone("efficientnet-b0")
    print(model)
    print(model(torch.zeros((1, 3, 128, 128)))[-1].shape)
