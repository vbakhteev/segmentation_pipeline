from abc import ABC, abstractmethod
from pathlib import Path
from typing import Callable

from torch.utils.data import Dataset


class BaseDataset(Dataset, ABC):
    """This class is an abstract base class (ABC) for datasets.

    To create a subclass, you need to implement the following four functions:
       -- <__init__>:                      initialize the class,
        first call BaseDataset.__init__(self, cfg, transforms).
       -- <__getitem>:                     get a data point.
       -- <__len>:                         defines size of dataset instead of __len__
       -- <prepare_data>:                  load train/val/tests split. Called in LightningModule.prepare_data
    """

    def __init__(self, cfg, transforms: Callable, is_train: bool):
        """Initialize the class; save the options in the class

        Parameters:
            cfg (Config) -- contains all the experiment parameters; needs to be an instance of EasyDict
            transforms -- preprocessing transformations of samples
        """
        self.samples_per_epoch = get_samples_per_epoch(cfg)
        self.root = Path(cfg.dataset.root)
        self.transforms = transforms
        self.is_train = is_train

    @abstractmethod
    def _getitem(self, index: int) -> dict:
        """Return a data point and its metadata information.

        Parameters:
            index - a random integer for data indexing
        Returns:
            a dictionary of data with their names. It usually contains the data itself and its metadata information.
        """
        pass

    @abstractmethod
    def _len(self) -> int:
        """Returns dataset size"""
        pass

    @staticmethod
    @abstractmethod
    def prepare_data(cfg) -> dict:
        """Read train/validation/test split or whatever required for preparation

        Returns:
            a dictionary of dictionaries that will be used for creation of dataset and dataloader
            {'train': {params...}
            'valid': {params...}
            'test': {params...}}

            params are arguments that will be passed in __init__()
        """
        pass

    def get_valid_index(self, index: int) -> int:
        return index % self._len()

    def __getitem__(self, index: int) -> dict:
        """Return a data point and its metadata information.

        Parameters:
            index - a random integer for data indexing
        Returns:
            a dictionary of data with their names. It usually contains the data itself and its metadata information.
        """
        if self.is_train:
            index = self.get_valid_index(index)

        return self._getitem(index)

    def __len__(self) -> int:
        """Return the number of images processed during one epoch"""
        if self.is_train:
            return self.samples_per_epoch
        else:
            return self._len()


def get_samples_per_epoch(cfg) -> int:
    steps_per_epoch = cfg.dataset.steps_per_epoch
    n_gpus = max(cfg.lightning.gpus * cfg.lightning.num_nodes, 1)
    batch_size = cfg.dataloader.batch_size

    return steps_per_epoch * n_gpus * batch_size
