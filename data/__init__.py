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

from data.heplers.transforms import get_transforms
from data.heplers.multitask_dataset import MultiTaskDataset, MultiTaskSampler
from data.heplers.utils import find_dataset_using_name


def get_dataloader(
    cfg,
    stage: str,
    transforms_: Optional[Callable] = None,
) -> DataLoader:
    if stage not in ("train", "valid", "test"):
        raise KeyError(f"{stage} dataloader not is not supported")

    is_train = stage == "train"

    datasets, dataset_ids, dataset_names = get_datasets(
        cfg=cfg,
        stage=stage,
        is_train=is_train,
        transforms_=transforms_,
    )
    dataset = MultiTaskDataset(datasets, dataset_ids, dataset_names)
    sampler = MultiTaskSampler(dataset, cfg.dataloader.batch_size)

    # len(loader) = steps_per_epoch * len(datasets)
    loader = DataLoader(
        dataset,
        drop_last=is_train,
        sampler=sampler if is_train else None,
        collate_fn=None,
        **cfg.dataloader,  # num_workers,batch_size,pin_memory
    )

    return loader


def get_samples_per_epoch(cfg):
    spe = cfg.datasets.steps_per_epoch
    bs = cfg.dataloader.batch_size
    n_gpus = max(cfg.lightning.gpus * cfg.lightning.num_nodes, 1)

    return spe * n_gpus * bs


def get_datasets(
    cfg, stage: str, is_train: bool, transforms_: Optional[Callable]
) -> tuple:
    samples_per_epoch = get_samples_per_epoch(cfg)

    datasets, dataset_ids, dataset_names = [], [], []
    for dataset_cfg in cfg.datasets.list:
        if stage == "train" and not dataset_cfg.use_to_train:
            continue
        if stage == "valid" and not dataset_cfg.use_to_validate:
            continue

        data_transforms = transforms_ or get_transforms(dataset_cfg)[stage]
        dataset_cls = find_dataset_using_name(dataset_cfg.name)
        dataset_params = dataset_cls.prepare_data(dataset_cfg)[stage]

        dataset = dataset_cls(
            root=dataset_cfg.root,
            samples_per_epoch=samples_per_epoch,
            transforms=data_transforms,
            is_train=is_train,
            **dataset_params,
        )

        datasets += [dataset]
        dataset_names += [dataset_cfg.name]
        dataset_ids += [dataset_cfg.id]

    return datasets, dataset_ids, dataset_names
