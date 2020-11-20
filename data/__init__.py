"""This package includes all the modules related to data loading and preprocessing
 To add a custom dataset class called 'dummy', you need to add a file called
 'dummy_dataset.py' and define a subclass 'DummyDataset' inherited from BaseDataset.
 You need to implement four functions:
   -- <__init__>:                     initialize the class,
   first call super().__init__(root, samples_per_epoch, transforms, is_train).
    -- <_getitem>:                     get a data point.
    -- <_len>:                         defines size of dataset instead of __len__.
    -- <prepare_data>:                 returns additional arguments to __init__.
See our template dataset class 'human_dataset.py' for more details.
"""

from typing import Callable, Optional

from torch.utils.data import DataLoader

from data.transforms import get_transforms
from .utils import find_dataset_using_name


def get_dataloader(
    cfg, stage: str, transforms_: Optional[Callable] = None
) -> DataLoader:
    if stage not in ("train", "valid", "test"):
        raise KeyError(f"{stage} dataloader not is not supported")

    is_train = stage == "train"
    transforms_ = transforms_ or get_transforms(cfg)[stage]
    samples_per_epoch = get_samples_per_epoch(cfg)

    dataset_cls = find_dataset_using_name(cfg.dataset.name)
    dataset_params = dataset_cls.prepare_data(cfg)[stage]
    dataset = dataset_cls(
        root=cfg.dataset.root,
        samples_per_epoch=samples_per_epoch,
        transforms=transforms_,
        is_train=is_train,
        **dataset_params,
    )

    loader = DataLoader(
        dataset,
        shuffle=is_train,
        drop_last=is_train,
        sampler=None,
        collate_fn=None,
        **cfg.dataloader,  # num_workers,batch_size,pin_memory
    )

    return loader


def get_samples_per_epoch(cfg):
    spe = cfg.dataset.steps_per_epoch
    bs = cfg.dataloader.batch_size
    n_gpus = max(cfg.lightning.gpus * cfg.lightning.num_nodes, 1)

    return spe * n_gpus * bs
