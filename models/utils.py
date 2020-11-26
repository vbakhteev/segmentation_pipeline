import torch
from torch import nn


def get_layers_by_dim(n_dim: int) -> dict:
    assert n_dim in (1, 2, 3)

    layers = {
        "batch_norm": (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d),
        "conv": (nn.Conv1d, nn.Conv2d, nn.Conv3d),
        "conv_transpose": (nn.ConvTranspose1d, nn.ConvTranspose2d, nn.ConvTranspose3d),
        "pad": (nn.ReflectionPad1d, nn.ReflectionPad2d, nn.ReplicationPad3d),
        "max_pool": (nn.MaxPool1d, nn.MaxPool2d, nn.MaxPool3d),
        "avg_pool": (nn.AvgPool1d, nn.AvgPool2d, nn.AvgPool3d),
        "adaptive_max_pool": (
            nn.AdaptiveMaxPool1d,
            nn.AdaptiveMaxPool2d,
            nn.AdaptiveMaxPool3d,
        ),
        "adaptive_avg_pool": (
            nn.AdaptiveAvgPool1d,
            nn.AdaptiveAvgPool2d,
            nn.AdaptiveAvgPool3d,
        ),
    }

    result = dict()
    for layer_name, choices in layers.items():
        if len(choices) >= n_dim:
            layer = choices[n_dim - 1]
        else:
            layer = None

        result[layer_name] = layer

    return result


def change_layers_dim(
    from_dim: int,
    to_dim: int,
    layer_names=("Conv", "BatchNorm", "MaxPool"),
):
    def decorator(function):
        def wrapper(*args, **kwargs):

            # Replace nn.Layer{from_dim}d to nn.Layer{to_dim}d
            memory = dict()
            for name in layer_names:
                name_to_d, name_from_d = name + f"{from_dim}d", name + f"{to_dim}d"
                memory[name_to_d] = getattr(nn, name_to_d)
                setattr(nn, name_to_d, getattr(nn, name_from_d))

            result = function(*args, **kwargs)

            # Return nn.Layer{from_dim}d backward
            for name_to_d, v in memory.items():
                setattr(nn, name_to_d, v)

            return result

        return wrapper

    return decorator


def change_num_channels_first_conv(
    model: nn.Module, n_dim: int, num_channels: int
) -> nn.Module:
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

        Conv = get_layers_by_dim(n_dim)["conv"]
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
