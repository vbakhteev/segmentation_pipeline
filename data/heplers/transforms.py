import pydoc

import albumentations as albu
import numpy as np
import torch
from albumentations.pytorch import ToTensor, ToTensorV2

from data.heplers.functional_transforms import resize_mask3d, resize_img3d


def get_transforms(dataset_cfg):
    pre_transforms = parse_transforms(dataset_cfg.pre_transforms)
    augmentations = parse_transforms(dataset_cfg.augmentations)
    post_transforms = parse_transforms(dataset_cfg.post_transforms)

    transforms = dict(
        train=pre_transforms + augmentations + post_transforms,
        valid=pre_transforms + post_transforms,
        test=pre_transforms + post_transforms,
    )
    for k, v in transforms.items():
        transforms[k] = albu.Compose(v)

    return transforms


def parse_transforms(transforms_cfg: list) -> list:
    transforms = []

    for transform in transforms_cfg:
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

    if cls is None:
        raise KeyError(f"Transform {name} is not supported")

    return cls(*args, **kwargs)


class ToTensor3D(albu.BasicTransform):
    """Convert image and mask to `torch.Tensor`."""

    def __init__(self, always_apply=True, p=1.0):
        super().__init__(always_apply=always_apply, p=p)

    @property
    def targets(self):
        return {"image": self.apply, "mask": self.apply_to_mask}

    def apply(self, img, **params):  # skipcq: PYL-W0613
        if len(img.shape) not in [3, 4]:
            raise ValueError("ToTensor3D only supports images in LHW or LHWC format")

        if len(img.shape) == 3:
            img = np.expand_dims(img, 3)

        return torch.from_numpy(img.transpose(3, 0, 1, 2))

    def apply_to_mask(self, mask, **params):  # skipcq: PYL-W0613
        if mask.ndim == 4:
            mask = mask.transpose(3, 0, 1, 2)
        return torch.from_numpy(mask)

    def get_params_dependent_on_targets(self, params):
        return {}


class Crop3d(albu.DualTransform):
    def __init__(self, size: tuple, always_apply=True, p=1.0):
        super().__init__(always_apply, p)
        self.size = size

    def apply(self, img, **params):
        # TODO improve cropping
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


class RandomFlip3d(albu.DualTransform):
    def __init__(self, axis: int, always_apply=False, p=0.5):
        super().__init__(always_apply, p)
        self.axis = axis

    def apply(self, img, **params):
        return np.flip(img, axis=self.axis)

    def apply_to_mask(self, img, **params):
        return np.flip(img, axis=self.axis)

    def get_transform_init_args_names(self):
        return ("axis",)


class RandomNoise3d(albu.ImageOnlyTransform):
    def __init__(self, mean: float = 0, std: float = 0.25, always_apply=False, p=1.0):
        super().__init__(always_apply, p)
        self.mean = mean
        self.std = std

    def apply(self, img, **params):
        noise = np.random.normal(loc=self.mean, scale=self.std, size=img.shape)
        img = img.astype(float) + noise
        return img

    def get_transform_init_args_names(self):
        return ("mean", "std")


if __name__ == "__main__":
    # For pycharm to not remove imports during imports optimization
    ToTensor()
    ToTensorV2()
