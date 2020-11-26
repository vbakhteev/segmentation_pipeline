from .resnet import get_resnet

resnets = [f"resnet{i}" for i in (18, 34, 50, 101, 152)]


def get_backbone_3d(model_name):
    if model_name in resnets:
        model = get_resnet(model_name)

    else:
        raise NotImplementedError(f"Backbone {model_name} is not supported")

    return model
