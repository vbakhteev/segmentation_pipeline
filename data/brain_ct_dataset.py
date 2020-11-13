from pathlib import Path
from typing import Iterable, Callable

from .base_dataset import BaseDataset


class BrainCTDataset(BaseDataset):
    """A dataset class for CT image dataset.
    It assumes that the directory `root` contains folders `images` and `masks`
    with all train, validation and test images/masks.
    """

    def __init__(self, cfg, transforms: Callable, filenames: list):
        """Initialize this dataset class.
        Parameters:
               cfg (Config) -- contains all the experiment parameters; needs to be an instance of EasyDict
               transforms -- preprocessing transformations of samples
               filenames -- list of filenames without extensions.
        """
        BaseDataset.__init__(self, cfg, transforms)

        filenames_set = set(filenames)
        self.imgs_paths = filter_paths(self.root / "images", filenames_set)
        self.masks_paths = filter_paths(self.root / "masks", filenames_set)

        assert len(self.imgs_paths) == len(self.masks_paths)

    def __getitem__(self, index: int) -> dict:
        """Returns a dictionary that contains image, mask and some metadata
        image (tensor) -- an image in the input domain
        mask (tensor) -- corresponding mask
        """
        image = None
        mask = None

        sample = self.transforms(image=image, mask=mask)
        image = sample["image"]
        mask = sample["mask"]

        return {"image": image, "mask": mask}


def filter_paths(directory: Path, filenames: Iterable) -> list:
    paths = directory.glob("*")
    valid_paths = sorted([p for p in paths if p.stem in filenames])
    return valid_paths
