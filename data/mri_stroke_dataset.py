from pathlib import Path
from typing import Union, Callable

import cv2
import torch
import numpy as np
from sklearn.model_selection import train_test_split

from data.heplers.base_dataset import BaseDataset


class MRIStrokeDataset(BaseDataset):
    """
    Data resides here: /mercurySOFS/Medical/MKDC_stroke_CT_MRI/1stBatchWithMasks
    """

    def __init__(
        self,
        root: Union[str, Path],
        samples_per_epoch: int,
        transforms: Callable,
        is_train: bool,
        patient_names: list,
    ):
        super().__init__(root, samples_per_epoch, transforms, is_train)
        self.paths = []
        for patient_path in self.root.iterdir():
            if patient_path.stem in patient_names:
                self.paths.extend(self.read_patient(patient_path))

    def _len(self):
        return len(self.paths)

    def _getitem(self, index: int) -> dict:
        """Returns a dictionary that contains image, mask and some metadata
        image (tensor) -- an image in the input domain
        mask (tensor) -- corresponding mask
        """
        img_path, mask_path = self.paths[index]
        image = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)[..., np.newaxis]
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if mask.max() == 255:
            mask = mask / 255

        sample = self.transforms(image=image, mask=mask)
        image = sample["image"].type(torch.float)
        mask = sample["mask"].type(torch.int64)

        return {"image": image, "mask": mask}

    @staticmethod
    def prepare_data(dataset_cfg=None, root=None) -> dict:
        """Read train/validation split
        Returns:
            a dictionary of data that will be used for creation of dataset and dataloader
        """
        root = root if dataset_cfg is None else dataset_cfg.root
        root = Path(root)
        dirnames = [p.stem for p in root.iterdir() if not p.stem.startswith(".")]

        train_filenames, valid_filenames = train_test_split(dirnames, random_state=42)

        result = dict(
            train={"patient_names": train_filenames},
            valid={"patient_names": valid_filenames},
        )
        return result

    @staticmethod
    def read_patient(patient_path):
        mask_dir = patient_path / "masks"
        img_dir = next(patient_path.glob("*DWI*"))
        filenames = mask_dir.glob("*.png")

        ret = []
        for name in filenames:
            img_path = img_dir / name.name
            mask_path = name
            ret += [(img_path, mask_path)]

        return ret
