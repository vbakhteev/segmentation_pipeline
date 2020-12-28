from pathlib import Path
from typing import Callable, Union

import pandas as pd
import numpy as np
import nibabel as nib
import torch
from sklearn.model_selection import GroupKFold

from data.heplers.base_dataset import BaseDataset


class AtlasDataset(BaseDataset):
    def __init__(
        self,
        root: Union[str, Path],
        samples_per_epoch: int,
        transforms: Callable,
        is_train: bool,
        df,
    ):
        super().__init__(root, samples_per_epoch, transforms, is_train)
        self.df = df.reset_index(drop=True)

    def _getitem(self, index: int) -> dict:
        """Returns a dictionary that contains image, mask and some metadata
        image (tensor) -- an image in the input domain
        mask (tensor) -- corresponding mask
        """
        row = self.df.iloc[index]
        subject_id = str(row["INDI Subject ID"]).zfill(6)
        sample_path = self.root / row["INDI Site ID"] / subject_id / row["Session"]
        image_path = sample_path / (subject_id + "_t1w_deface_stx.nii.gz")
        mask_paths = sample_path.glob("*_LesionSmooth*stx.nii.gz")

        image = nib.load(image_path).get_fdata() - 30
        image = image / 35
        masks = [nib.load(p).get_fdata() for p in mask_paths]
        mask = (np.stack(masks).sum(0) != 0).astype(int)

        sample = self.transforms(image=image, mask=mask)
        image = sample["image"].float()
        mask = sample["mask"].type(torch.int64)

        return {"image": image, "mask_atlas": mask}

    def _len(self):
        return len(self.df)

    @staticmethod
    def prepare_data(dataset_cfg) -> dict:
        root = Path(dataset_cfg.root)
        df = pd.read_csv(root / "ATLAS_Meta-Data_Release_1.1_standard_mni.csv")
        df["Session"] = df["Session"].apply(lambda s: s.replace(" ", ""))

        kfold = GroupKFold(n_splits=5)
        train_idxs, valid_idxs = next(
            iter(kfold.split(df, groups=df["INDI Subject ID"]))
        )

        df_train = df.iloc[train_idxs]
        df_valid = df.iloc[valid_idxs]
        result = dict(
            train={"df": df_train},
            valid={"df": df_valid},
        )

        return result
