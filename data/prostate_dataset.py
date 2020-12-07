from pathlib import Path
from typing import Iterable, Callable, Union

import torch
import nibabel as nib
from sklearn.model_selection import train_test_split

from data.heplers.base_dataset import BaseDataset


class ProstateDataset(BaseDataset):
    """
    Data resides at http://medicaldecathlon.com/index.html
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
        self.imgs_paths = filter_paths(self.root / "imagesTr", filenames_set)
        self.masks_paths = filter_paths(self.root / "labelsTr", filenames_set)

        assert len(self.imgs_paths) == len(self.masks_paths)

    def _getitem(self, index: int) -> dict:
        """Returns a dictionary that contains image, mask and some metadata
        image (tensor) -- an image in the input domain
        mask (tensor) -- corresponding mask
        """
        image = nib.load(self.imgs_paths[index]).get_fdata()[..., 0] / 3370.0
        mask = nib.load(self.masks_paths[index]).get_fdata()
        sample = self.transforms(image=image, mask=mask)
        image = sample["image"].type(torch.float)
        mask = sample["mask"].type(torch.int64)

        return {"image": image, "mask_prostate": mask}

    def _len(self):
        return len(self.imgs_paths)

    @staticmethod
    def prepare_data(dataset_cfg=None) -> dict:
        """Read train/validation split
        Returns:
            a dictionary of data that will be used for creation of dataset and dataloader
        """
        root = Path(dataset_cfg.root)
        filenames = [p.stem for p in (root / "imagesTr").glob("[!._]*.nii*")]
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
