from abc import ABC, abstractmethod
from pathlib import Path
from typing import Callable, Union

from torch.utils.data import Dataset


class BaseDataset(Dataset, ABC):
    """This class is an abstract base class (ABC) for datasets.

    To create a subclass, you need to implement the following four functions:
       -- <__init__>:                     initialize the class,
        first call super().__init__(root, samples_per_epoch, transforms, is_train).
       -- <_getitem>:                     get a data point.
       -- <_len>:                         defines size of dataset instead of __len__.
       -- <prepare_data>:                 returns additional arguments to __init__.
    """

    def __init__(
        self,
        root: Union[str, Path],
        samples_per_epoch: int,
        transforms: Callable,
        is_train: bool,
    ):
        """Initialize the class; save the options in the class

        Parameters:
            root -- str or Path instance to the root of data
            samples_per_epoch -- how many samples in one epoch train dataset has
            transforms -- preprocessing transformations of samples
            is_train -- boolean
        """
        self.root = Path(root)
        self.samples_per_epoch = samples_per_epoch
        self.transforms = transforms
        self.is_train = is_train

    @abstractmethod
    def _getitem(self, index: int) -> dict:
        """Return a data point and its metadata information.

        Parameters:
            index - a random integer for data indexing
        Returns:
            a dictionary of data with their names.
            It usually contains the data itself and its metadata information.
        """
        pass

    @abstractmethod
    def _len(self) -> int:
        """Returns dataset size"""
        pass

    @staticmethod
    @abstractmethod
    def prepare_data(dataset_cfg) -> dict:
        """Read train/validation/test split or whatever required for your dataset

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
            a dictionary of data with their names.
            It usually contains the data itself and its metadata information.
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
