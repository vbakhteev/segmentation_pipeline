import sys,os
import cv2
from sklearn.model_selection import train_test_split
from data.base_dataset import BaseDataset
from pathlib import Path


class Mri1stBatchDataset(BaseDataset):
    '''
    Data resides here: /mercurySOFS/Medical/MKDC_stroke_CT_MRI/1stBatchWithMasks
    '''
    def __init__(self, root, samples_per_epoch, is_train, patient_names, transforms=None, augmentations=None):
        BaseDataset.__init__(self, root, samples_per_epoch, transforms, is_train)
        self.augmentations = augmentations
        self.paths = []
        for name in self.root.iterdir():
            if name.stem in patient_names:
                self.paths.extend(self.read_patient(name))

    def _len(self):
        return len(self.paths)

    def _getitem(self, index: int) -> dict:
        """Returns a dictionary that contains image, mask and some metadata
        image (tensor) -- an image in the input domain
        mask (tensor) -- corresponding mask
        """
        img_path, mask_path = self.paths[index]
        image = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)

        if self.augmentations is not None:
            sample = self.augmentations(image=image, mask=mask)
            image = sample["image"]
            mask = sample["mask"]

        if self.transforms is not None:
            sample = self.transforms(image=image, mask=mask)
            image = sample["image"]
            mask = sample["mask"]

        return {"image": image, "mask": mask}

    @staticmethod
    def prepare_data(root=None, cfg=None) -> dict:
        """Read train/validation split
        Returns:
            a dictionary of data that will be used for creation of dataset and dataloader
        """
        if cfg is not None:
            root = Path(cfg.dataset.root)
        else:
            root = Path(root)
        dirnames = [p.stem for p in root.iterdir() if not p.stem.startswith('.')]

        train_filenames, valid_filenames = train_test_split(dirnames)

        result = dict(
            train={"patient_names": train_filenames},
            valid={"patient_names": valid_filenames},
        )
        return result

    @staticmethod
    def read_patient(root):
        root = Path(root)
        mask_dir = root / 'masks'
        img_dir = next(root.glob('*DWI*'))
        filenames = mask_dir.glob('*.png')
        ret = []
        for name in filenames:
            img_path = img_dir / name.name
            mask_path = name
            ret.append((img_path, mask_path))
        return ret