from .backbones_2d import get_backbone_2d
from .backbones_3d import get_backbone_3d


def get_encoder(encoder_name, n_dim, num_channels):
    if n_dim == 2:
        model = get_backbone_2d(model_name=encoder_name)
    elif n_dim == 3:
        model = get_backbone_3d(model_name=encoder_name)
    else:
        raise NotImplementedError(f"Models with n_dim={n_dim} are not supported")

    model.change_in_channels(num_channels=num_channels, n_dim=n_dim)
    return model
