"""This package includes all the modules related to data loading and preprocessing
 To add a custom dataset class called 'dummy', you need to add a file called
 'dummy_dataset.py' and define a subclass 'DummyDataset' inherited from BaseDataset.
 You need to implement four functions:
    -- <__init__>:                      initialize the class, first call BaseDataset.__init__(self, opt).
    -- <__len__>:                       return the size of dataset.
    -- <__getitem__>:                   get a data point from data loader.
    -- <modify_commandline_options>:    (optionally) add dataset-specific options and set default options.
Now you can use the dataset class by specifying flag '--dataset_mode dummy'.
See our template dataset class 'template_dataset.py' for more details.
"""

from torch.utils.data import DataLoader

from data.transforms import get_transforms
from .utils import find_dataset_using_name


def get_dataloader(cfg, stage: str, **kwargs) -> DataLoader:
    """Returns dataloader by given config and stage
    stage: [train | valid | test]
    """
    dataset_cls = find_dataset_using_name(cfg.dataset.name)
    transforms_ = get_transforms(cfg)
    transforms_ = transforms_[stage]
    is_train = stage == "train"

    dataset = dataset_cls(
        cfg=cfg,
        transforms=transforms_,
        is_train=is_train,
        **kwargs,
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
