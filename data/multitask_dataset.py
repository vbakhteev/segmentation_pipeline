import random

import numpy as np
import torch
from torch.utils.data import Dataset


class MultiTaskDataset(Dataset):
    def __init__(self, datasets: list, dataset_ids: list, dataset_names: list):
        self.datasets = datasets
        self.dataset_ids = dataset_ids
        self.dataset_names = dataset_names

        lens = [len(dataset) for dataset in self.datasets]
        self.cumsum = np.cumsum(lens, dtype=np.int)

    def __getitem__(self, index: int) -> dict:
        dataset, dataset_id, dataset_name, index = self.idx2dataset(index)
        sample = dataset[index]
        sample["dataset_id"] = dataset_id
        sample["dataset_name"] = dataset_name

        return sample

    def __len__(self):
        return self.cumsum[-1]

    def idx2dataset(self, index: int) -> tuple:
        result_index = dataset_idx = -1
        prev_sum = 0

        for dataset_idx, s in enumerate(self.cumsum):
            if index < s:
                result_index = index - prev_sum
                break
            prev_sum = s

        dataset = self.datasets[dataset_idx]
        dataset_id = self.dataset_ids[dataset_idx]
        dataset_name = self.dataset_names[dataset_idx]
        return dataset, dataset_id, dataset_name, result_index


class MultiTaskSampler(torch.utils.data.sampler.Sampler):
    def __init__(self, multitask_dataset, batch_size):
        self.bs = batch_size
        self.num_samples = len(multitask_dataset)

        cumsum = 0
        self.indices = []
        for d in multitask_dataset.datasets:
            self.indices.append(list(range(cumsum, cumsum + len(d))))
            cumsum += len(d)

    def __iter__(self):
        shuffled_ids = [random.sample(ids, len(ids)) for ids in self.indices]

        result = []
        for batch_i in range(int(self.num_samples / self.bs)):
            for dataset_id in range(len(self.indices)):
                dataset_indices = shuffled_ids[dataset_id]
                result += dataset_indices[self.bs * batch_i : self.bs * (batch_i + 1)]

        return (i for i in result)

    def __len__(self):
        return self.num_samples
