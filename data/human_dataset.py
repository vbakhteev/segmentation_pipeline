from pathlib import Path
from typing import Iterable, Callable

import cv2
from sklearn.model_selection import train_test_split

from .base_dataset import BaseDataset


class HumanDataset(BaseDataset):
    """A dataset class for Human segmentation (for debugging purposes).
    https://github.com/VikramShenoy97/Human-Segmentation-Dataset
    It assumes that the directory `root` contains folders `Training_Images` and `Ground_Truth`
    with all train, validation and test images/masks.
    """

    def __init__(self, cfg, transforms: Callable, is_train: bool, filenames: list):
        """Initialize this dataset class.
        Parameters:
               cfg (Config) -- contains all the experiment parameters; needs to be an instance of EasyDict
               transforms -- preprocessing transformations of samples
               filenames -- list of filenames without extensions.
        """
        BaseDataset.__init__(self, cfg, transforms)

        self.is_train = is_train

        filenames_set = set(filenames)
        self.imgs_paths = filter_paths(self.root / "Training_Images", filenames_set)
        self.masks_paths = filter_paths(self.root / "Ground_Truth", filenames_set)

        assert len(self.imgs_paths) == len(self.masks_paths)

    def __getitem__(self, index: int) -> dict:
        """Returns a dictionary that contains image, mask and some metadata
        image (tensor) -- an image in the input domain
        mask (tensor) -- corresponding mask
        """
        if self.is_train:
            index = self.get_valid_index(index, len(self.imgs_paths))

        image = cv2.imread(str(self.imgs_paths[index]))[:, :, ::-1].copy()
        mask = cv2.imread(str(self.masks_paths[index]), 0)

        sample = self.transforms(image=image, mask=mask)
        image = sample["image"]
        mask = sample["mask"][0]

        return {"image": image, "mask": mask}

    def __len__(self):
        if self.is_train:
            return super().__len__()

        return len(self.imgs_paths)

    @staticmethod
    def prepare_data(cfg) -> dict:
        """Read train/validation split

        Returns:
            a dictionary of data that will be used for creation of dataset and dataloader
        """
        root = Path(cfg.dataset.root)
        filenames = [p.stem for p in (root / "Training_Images").glob("*.jpg")]
        train_filenames, valid_filenames = train_test_split(
            filenames, random_state=cfg.defaults.seed
        )

        return dict(train_filenames=train_filenames, valid_filenames=valid_filenames)


def filter_paths(directory: Path, filenames: Iterable) -> list:
    paths = directory.glob("*")
    valid_paths = sorted([p for p in paths if p.stem in filenames])
    return valid_paths
