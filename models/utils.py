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


def freeze(model):
    for param in model.parameters():
        param.requires_grad = False


def unfreeze(model):
    for param in model.parameters():
        param.requires_grad = True


def load_state_dict(model, path, soft=False):
    checkpoint = torch.load(path, map_location="cpu")
    if "pytorch-lightning_version" in checkpoint:
        checkpoint = filter_lightning_stuff(checkpoint)

    if soft:
        soft_load_state_dict(model, checkpoint)
    else:
        model.load_state_dict(checkpoint)


def filter_lightning_stuff(checkpoint):
    checkpoint = checkpoint["state_dict"]
    checkpoint = {k.replace("model.", "", 1): v for k, v in checkpoint.items()}
    return checkpoint


def soft_load_state_dict(model, state_dict):
    model_state = model.state_dict()

    not_loaded_params = []
    for name, param in state_dict.items():
        if name.startswith("module."):
            name = name[7:]

        if name not in model_state or model_state[name].shape != param.shape:
            not_loaded_params += [name]
            continue

        if isinstance(param, nn.Parameter):
            # backwards compatibility for serialized parameters
            param = param.data
        model_state[name].copy_(param)

    if len(not_loaded_params):
        print(
            "WARNING: following params couldn't be loaded into model:",
            not_loaded_params,
        )
