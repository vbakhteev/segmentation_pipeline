import pydoc

import numpy as np
import albumentations as albu
from albumentations.pytorch import ToTensor, ToTensorV2

from .functional_transforms import resize_mask3d, resize_img3d


def get_transforms(cfg):
    pre_transforms = parse_transforms(cfg.dataset.pre_transforms)
    augmentations = parse_transforms(cfg.dataset.augmentations)
    post_transforms = parse_transforms(cfg.dataset.post_transforms)

    transforms = dict(
        train=pre_transforms + augmentations + post_transforms,
        valid=pre_transforms + post_transforms,
        test=pre_transforms + post_transforms,
    )
    for k, v in transforms.items():
        transforms[k] = albu.Compose(v)

    return transforms


def parse_transforms(cfg) -> list:
    transforms = []

    for transform in cfg:
        if transform.name in ("Compose", "OneOf", "OneOrOther"):
            # get inner list of transforms
            inner_transforms = parse_transforms(transform.list)
            params = {k: v for k, v in transform.items() if k not in ("name", "list")}
            transform_obj = get_albu_object(transform.name, inner_transforms, **params)

        else:
            params = {k: v for k, v in transform.items() if k not in ("name", "list")}
            transform_obj = get_albu_object(transform.name, **params)

        transforms += [transform_obj]

    return transforms


def get_albu_object(name, *args, **kwargs):
    if name in globals():
        cls = globals()[name]
    else:
        cls_path = "albumentations." + name
        cls = pydoc.locate(cls_path)

    return cls(*args, **kwargs)


class Crop3d(albu.DualTransform):
    def __init__(self, size: tuple, always_apply=False, p=1.0):
        super().__init__(always_apply, p)
        self.size = size

    def apply(self, img, **params):
        return img[: self.size[0], : self.size[1], : self.size[2]]

    def apply_to_mask(self, mask, **params):
        return mask[: self.size[0], : self.size[1], : self.size[2]]

    def get_transform_init_args_names(self):
        return ("size",)


class Resize3d(albu.DualTransform):
    def __init__(self, size: tuple, always_apply=False, p=1.0):
        super().__init__(always_apply, p)
        self.size = np.array(size)

    def apply(self, img, **params):
        return resize_img3d(img, self.size)

    def apply_to_mask(self, mask, **params):
        return resize_mask3d(mask, self.size)

    def get_transform_init_args_names(self):
        return ("size",)


if __name__ == "__main__":
    # For pycharm to not remove imports during imports optimization
    ToTensor()
    ToTensorV2()
