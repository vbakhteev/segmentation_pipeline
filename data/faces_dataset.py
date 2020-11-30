from pathlib import Path
from typing import Callable, Union

import PIL
import cv2
import numpy as np
import torch

from .base_dataset import BaseDataset


class FacesDataset(BaseDataset):
    """A dataset class for faces of human segmentation (for debugging purposes).
    https://github.com/massimomauro/FASSEG-repository
    """

    def __init__(
        self,
        root: Union[str, Path],
        samples_per_epoch: int,
        transforms: Callable,
        is_train: bool,
        prefix_dir: str,
    ):
        super().__init__(root, samples_per_epoch, transforms, is_train)
        imgs_dir = self.root / "V1" / f"{prefix_dir}_RGB"
        masks_dir = self.root / "V1" / f"{prefix_dir}_Labels"
        self.imgs_paths = [p for p in imgs_dir.glob("*.bmp")]
        self.masks_paths = [p for p in masks_dir.glob("*.bmp")]

        assert len(self.imgs_paths) == len(self.masks_paths)

    def _getitem(self, index: int) -> dict:
        """Returns a dictionary that contains image, mask and some metadata
        image (tensor) -- an image in the input domain
        mask (tensor) -- corresponding mask
        """
        image = np.array(PIL.Image.open(self.imgs_paths[index]))
        mask = np.array(PIL.Image.open(self.masks_paths[index]))
        mask = correct_mask(mask)

        sample = self.transforms(image=image, mask=mask)
        image = sample["image"]
        mask = sample["mask"].type(torch.int64)

        return {"image": image, "mask_face": mask}

    def _len(self):
        return len(self.imgs_paths)

    @staticmethod
    def prepare_data(dataset_cfg) -> dict:
        result = dict(
            train={"prefix_dir": "Train"},
            valid={"prefix_dir": "Test"},
        )
        return result


def correct_mask(mask):
    color_bounds = [
        ([110, 250, 250], [130, 255, 255]),  # Red BG
        ([85, 250, 50], [95, 255, 255]),  # yellow skin
        ([110, 250, 120], [130, 260, 234]),  # brown hairs
    ]

    mask_hsv = cv2.cvtColor(mask, cv2.COLOR_BGR2HSV)
    res = []

    for class_id, (lower, upper) in enumerate(color_bounds):
        m = cv2.inRange(mask_hsv, np.array(lower), np.array(upper))
        res += [m]

    res += [np.ones(mask.shape[:2])]
    mask = np.stack(res).argmax(axis=0)

    return mask
