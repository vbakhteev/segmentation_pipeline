import json
from pathlib import Path
from typing import Callable, Union, List

import pandas as pd
import nibabel as nib
import torch
from sklearn.model_selection import GroupKFold

from data.heplers.base_dataset import BaseDataset


class IslesDataset(BaseDataset):
    def __init__(
        self,
        root: Union[str, Path],
        samples_per_epoch: int,
        transforms: Callable,
        is_train: bool,
        image_paths: List[Path],
        mask_paths: List[Path],
        image_type: str,
    ):
        super().__init__(root, samples_per_epoch, transforms, is_train)
        self.image_paths = image_paths
        self.mask_paths = mask_paths
        self.image_type = image_type

    def _getitem(self, index: int) -> dict:
        """Returns a dictionary that contains image, mask and some metadata
        image (tensor) -- an image in the input domain
        mask (tensor) -- corresponding mask
        """
        img_path = self.image_paths[index]
        mask_path = self.mask_paths[index]
        image = nib.load(img_path).get_fdata()
        mask = nib.load(mask_path).get_fdata()

        sample = self.transforms(image=image, mask=mask)
        image = sample["image"].float()
        mask = sample["mask"].type(torch.int64)

        return {"image": image, f"mask_isles_{self.image_type}": mask}

    def _len(self):
        return len(self.image_paths)

    @staticmethod
    def prepare_data(dataset_cfg) -> dict:
        root = Path(dataset_cfg.root)
        train_path = root / "TRAINING"
        image_type = dataset_cfg.image_type

        df = get_patient_df(train_path)
        kfold = GroupKFold(n_splits=5)
        for tr_idxs, val_idxs in kfold.split(df, groups=df["patient_id"]):
            df_train = df.loc[tr_idxs].reset_index()
            df_valid = df.loc[val_idxs].reset_index()
            break

        train_image_paths, train_mask_paths = get_paths(
            cases=df_train["case"], data_dir=train_path, image_type=image_type
        )
        valid_image_paths, valid_mask_paths = get_paths(
            cases=df_valid["case"], data_dir=train_path, image_type=image_type
        )

        result = dict(
            train={
                "image_paths": train_image_paths,
                "mask_paths": train_mask_paths,
                "image_type": image_type,
            },
            valid={
                "image_paths": valid_image_paths,
                "mask_paths": valid_mask_paths,
                "image_type": image_type,
            },
        )

        return result


def get_paths(cases: List[str], data_dir: Path, image_type: str):
    image_paths, mask_paths = [], []

    for case in cases:
        case_path = data_dir / case
        image_path = list(case_path.glob(f"SMIR.Brain.XX.O.{image_type}.*/*.nii"))[0]
        mask_path = list(case_path.glob("SMIR.Brain.XX.O.OT.*/*.nii"))[0]

        image_paths += [image_path]
        mask_paths += [mask_path]

    return image_paths, mask_paths


def get_patient_df(data_dir):
    case_dirs = sorted(
        list(data_dir.glob("case_*")), key=lambda p: int(p.stem.split("_")[1])
    )

    info = []
    for case_dir in case_dirs:
        json_path = list(case_dir.glob("SMIR.Brain.XX.O.OT.*/*.json"))[0]
        with open(json_path) as f:
            data = json.load(f)

        study_id = data["description"].split(" - ")[2]
        user_id = study_id.split("_")[1]
        letter = study_id.split("_")[2]

        info.append((case_dir.stem, user_id, letter))

    return pd.DataFrame(info, columns=["case", "patient_id", "letter"])
