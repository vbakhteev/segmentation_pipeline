import torch
from torch import nn

from models.utils import get_layers_by_dim
from .backbones_2d import get_backbone_2d
from .backbones_3d import get_backbone_3d


def get_encoder(encoder_name, n_dim, num_channels):
    if n_dim == 2:
        model = get_backbone_2d(model_name=encoder_name)
    elif n_dim == 3:
        model = get_backbone_3d(model_name=encoder_name)
    else:
        raise NotImplementedError(f"Models with n_dim={n_dim} are not supported")

    if num_channels == 3:
        pass
    elif num_channels == 1:
        named_conv = next(model.named_parameters())
        name = named_conv[0].split(".")[:-1]
        w = named_conv[1]  # (out_filters, 3, k, k) or (out_filters, 3, k, k, k)

        nn_module = model
        for n in name[:-1]:
            nn_module = getattr(nn_module, n)
        first_conv = next(nn_module.children())

        Conv = get_layers_by_dim(n_dim)
        setattr(
            nn_module,
            name[-1],
            Conv(
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
        raise NotImplementedError(f"Num channels={num_channels} is not implemented")

    return model
