from abc import ABC, abstractmethod
from pathlib import Path
from typing import Callable

from torch.utils.data import Dataset


class BaseDataset(Dataset, ABC):
    """This class is an abstract base class (ABC) for datasets.

    To create a subclass, you need to implement the following four functions:
       -- <__init__>:                      initialize the class,
        first call BaseDataset.__init__(self, cfg, transforms).
       -- <__getitem__>:                   get a data point.
    """

    def __init__(self, cfg, transforms: Callable):
        """Initialize the class; save the options in the class

        Parameters:
            cfg (Config) -- contains all the experiment parameters; needs to be an instance of EasyDict
            transforms -- preprocessing transformations of samples
        """
        self.samples_per_epoch = get_samples_per_epoch(cfg)
        self.root = Path(cfg.dataset.root)
        self.transforms = transforms

    @abstractmethod
    def __getitem__(self, index: int) -> dict:
        """Return a data point and its metadata information.

        Parameters:
            index - a random integer for data indexing
        Returns:
            a dictionary of data with their names. It usually contains the data itself and its metadata information.
        """
        pass

    def __len__(self) -> int:
        """Return the number of images processed during one epoch"""
        return self.samples_per_epoch


def get_samples_per_epoch(cfg) -> int:
    steps_per_epoch = cfg.dataset.steps_per_epoch
    n_gpus = max(cfg.lightning.gpus, 1) * max(cfg.lightning.num_nodes, 1)
    batch_size = cfg.dataloader.batch_size

    return steps_per_epoch * n_gpus * batch_size
