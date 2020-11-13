from torch import nn

from utils import dict_remove_key, object_from_dict


def get_criterion(cfg_criterion):
    """Returns loss function by given config
    Available options:
    pytorch_toolbelt.losses.DiceLoss
    pytorch_toolbelt.losses.FocalLoss
    pytorch_toolbelt.losses.BinaryFocalLoss
    pytorch_toolbelt.losses.JaccardLoss
    pytorch_toolbelt.losses.LovaszLoss
    pytorch_toolbelt.losses.BinaryLovaszLoss
    pytorch_toolbelt.losses.SoftBCEWithLogitsLoss
    pytorch_toolbelt.losses.SoftCrossEntropyLoss
    pytorch_toolbelt.losses.WingLoss
    pytorch_toolbelt.losses.JointLoss
    MyCustomLoss
    Also, losses defined below and from other packages
    ----------------------------
    Example config of joint loss:
    criterion:
      cls: 'pytorch_toolbelt.losses.JointLoss'
      first_weight: 1.0
      second_weight: 2.0
      first:
        cls: 'pytorch_toolbelt.losses.DiceLoss'
      second:
        cls: 'pytorch_toolbelt.losses.FocalLoss'
        gamma: 3
    ----------------------------
    """
    if cfg_criterion.cls == "pytorch_toolbelt.losses.JointLoss":
        first = get_criterion(cfg_criterion.first)
        second = get_criterion(cfg_criterion.second)
        cfg_criterion = dict_remove_key(cfg_criterion, "first")
        cfg_criterion = dict_remove_key(cfg_criterion, "second")

        return object_from_dict(cfg_criterion, first=first, second=second)

    # Criterion from external package
    elif "." in cfg_criterion.cls:
        return object_from_dict(cfg_criterion)

    # Criterion from this file
    else:
        cls_name = cfg_criterion.pop("cls")
        cls = globals().get(cls_name)
        if cls is None:
            raise NotImplementedError(f"Criterion {cls_name} is not implemented")

        return cls(**cfg_criterion)


class MyCustomLoss(nn.Module):
    def forward(self, logits, target):
        return 0
