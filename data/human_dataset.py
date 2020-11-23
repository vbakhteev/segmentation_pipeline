from pathlib import Path
from typing import Iterable, Callable, Union

import cv2
from sklearn.model_selection import train_test_split

from .base_dataset import BaseDataset
from .utils import read_2d_img_cv2


class HumanDataset(BaseDataset):
    """A dataset class for Human segmentation (for debugging purposes).
    https://github.com/VikramShenoy97/Human-Segmentation-Dataset
    It assumes that the directory `root` contains folders `Training_Images` and `Ground_Truth`
    with all train, validation and test images/masks.
    """

    def __init__(
        self,
        root: Union[str, Path],
        samples_per_epoch: int,
        transforms: Callable,
        is_train: bool,
        filenames: list,
    ):
        super().__init__(root, samples_per_epoch, transforms, is_train)

        filenames_set = set(filenames)
        self.imgs_paths = filter_paths(self.root / "Training_Images", filenames_set)
        self.masks_paths = filter_paths(self.root / "Ground_Truth", filenames_set)

        assert len(self.imgs_paths) == len(self.masks_paths)

    def _getitem(self, index: int) -> dict:
        """Returns a dictionary that contains image, mask and some metadata
        image (tensor) -- an image in the input domain
        mask (tensor) -- corresponding mask
        """
        image = read_2d_img_cv2(self.imgs_paths[index])
        mask = cv2.imread(str(self.masks_paths[index]), 0)

        sample = self.transforms(image=image, mask=mask)
        image = sample["image"]
        mask = sample["mask"][0]

        return {"image": image, "mask": mask}

    def _len(self):
        return len(self.imgs_paths)

    @staticmethod
    def prepare_data(dataset_cfg) -> dict:
        """Read train/validation split
        Returns:
            a dictionary of data that will be used for creation of dataset and dataloader
        """
        root = Path(dataset_cfg.root)
        filenames = [p.stem for p in (root / "Training_Images").glob("*.jpg")]
        train_filenames, valid_filenames = train_test_split(filenames, random_state=42)

        result = dict(
            train={"filenames": train_filenames},
            valid={"filenames": valid_filenames},
        )
        return result


def filter_paths(directory: Path, filenames: Iterable) -> list:
    paths = directory.glob("*")
    valid_paths = sorted([p for p in paths if p.stem in filenames])
    return valid_paths
