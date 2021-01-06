import timm

from .se_resnext import get_seresnext
from .timm_models import get_timm_model


timm_models = timm.list_models()

se_resexts = ["se_resnext50_32x4d", "se_resnext101_32x4d"]


def get_backbone_2d(model_name):
    if model_name in timm_models:
        model = get_timm_model(model_name)

    elif model_name in se_resexts:
        model = get_seresnext(model_name)

    else:
        raise NotImplementedError("Backbone {} is not supported".format(model_name))

    return model
