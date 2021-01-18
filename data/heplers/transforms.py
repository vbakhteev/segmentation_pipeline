import pydoc

import cv2
import albumentations as albu
import numpy as np
import torch
from skimage import exposure
from albumentations.pytorch import ToTensor, ToTensorV2
from albumentations.augmentations import functional as F
from albumentations.augmentations import bbox_utils as bbu

from data.heplers.functional_transforms import resize_mask3d, resize_img3d


__all__ = [
    "get_transforms",
    "EqualizeHist",
    "EqualizeAdaptHist",
    "ToTensor3D",
    "Crop3d",
    "Resize3d",
    "RandomFlip3d",
    "RandomNoise3d",
    "PadToSquare",
]


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


class EqualizeHist(albu.ImageOnlyTransform):
    def __init__(self, always_apply=True, p=1.0):
        super().__init__(always_apply=always_apply, p=p)

    def apply(self, img, **params):
        return exposure.equalize_hist(img)


class EqualizeAdaptHist(albu.ImageOnlyTransform):
    def __init__(self, always_apply=True, p=1.0):
        super().__init__(always_apply=always_apply, p=p)

    def apply(self, img, **params):
        img -= img.min()
        img /= img.max()
        return exposure.equalize_adapthist(img)


class PadToSquare(albu.DualTransform):
    """Pad side of the image / max if side is less than desired number.
    Args:
        border_mode (OpenCV flag): OpenCV border mode.
        value (int, float, list of int, list of float): padding value if border_mode is cv2.BORDER_CONSTANT.
        mask_value (int, float,
                    list of int,
                    list of float): padding value for mask if border_mode is cv2.BORDER_CONSTANT.
        p (float): probability of applying the transform. Default: 1.0.
    Targets:
        image, mask, bbox, keypoints
    Image types:
        uint8, float32
    """

    def __init__(
        self,
        border_mode=cv2.BORDER_CONSTANT,
        value=0,
        mask_value=0,
        always_apply=True,
        p=1.0,
    ):
        super().__init__(always_apply, p)
        self.border_mode = border_mode
        self.value = value
        self.mask_value = mask_value

    def update_params(self, params, **kwargs):
        params = super().update_params(params, **kwargs)
        rows = params["rows"]
        cols = params["cols"]

        if rows < cols:
            h_pad_top = int((cols - rows) / 2.0)
            h_pad_bottom = cols - rows - h_pad_top
        else:
            h_pad_top = 0
            h_pad_bottom = 0

        if cols < rows:
            w_pad_left = int((rows - cols) / 2.0)
            w_pad_right = rows - cols - w_pad_left
        else:
            w_pad_left = 0
            w_pad_right = 0

        params.update(
            {"pad_top": h_pad_top, "pad_bottom": h_pad_bottom, "pad_left": w_pad_left, "pad_right": w_pad_right}
        )
        return params

    def apply(self, img, pad_top=0, pad_bottom=0, pad_left=0, pad_right=0, **params):
        return F.pad_with_params(
            img, pad_top, pad_bottom, pad_left, pad_right, border_mode=self.border_mode, value=self.value
        )

    def apply_to_mask(self, img, pad_top=0, pad_bottom=0, pad_left=0, pad_right=0, **params):
        return F.pad_with_params(
            img, pad_top, pad_bottom, pad_left, pad_right, border_mode=self.border_mode, value=self.mask_value
        )

    def apply_to_bbox(self, bbox, pad_top=0, pad_bottom=0, pad_left=0, pad_right=0, rows=0, cols=0, **params):
        x_min, y_min, x_max, y_max = bbu.denormalize_bbox(bbox, rows, cols)
        bbox = x_min + pad_left, y_min + pad_top, x_max + pad_left, y_max + pad_top
        return bbu.normalize_bbox(bbox, rows + pad_top + pad_bottom, cols + pad_left + pad_right)

    # skipcq: PYL-W0613
    def apply_to_keypoint(self, keypoint, pad_top=0, pad_bottom=0, pad_left=0, pad_right=0, **params):
        x, y, angle, scale = keypoint
        return x + pad_left, y + pad_top, angle, scale

    def get_transform_init_args_names(self):
        return (
            "border_mode",
            "value",
            "mask_value",
        )


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
