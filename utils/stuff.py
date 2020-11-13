import copy
import os
import pydoc
import random

import numpy as np
import torch


def set_deterministic(seed=322, precision=10):
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.set_printoptions(precision)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def object_from_dict(d: dict, **default_kwargs):
    kwargs = d.copy()
    cls_path = kwargs.pop("cls")
    for k, v in default_kwargs.items():
        if k not in kwargs:
            kwargs[k] = v

    cls = pydoc.locate(cls_path)
    obj = cls(**kwargs)  # type: ignore
    return obj


def dict_remove_key(d, key):
    c = copy.deepcopy(d)
    del c[key]
    return c
