import logging
import os

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, Sampler

logger = logging.getLogger("DeepAR.Data")


class TrainDataset(Dataset):
    def __init__(self, data_path, data_name, num_class):
        # Load training dataset from Numpy dump - might not be able to do this
        # if dataset gets large
        self.data = np.load(os.path.join(data_path, f"train_data_{data_name}.npy"))
        self.label = np.load(os.path.join(data_path, f"train_label_{data_name}.npy"))
        self.train_len = self.data.shape[0]
        logger.info(f"train_len: {self.train_len}")
        logger.info(f"building train dataset from {data_path}...")

    def __len__(self):
        return self.train_len

    def __getitem__(self, index):
        return (
            self.data[index, :, :-1],
            int(self.data[index, 0, -1]),
            self.label[index],
        )


class TestDataset(Dataset):
    def __init__(self, data_path, data_name, num_class):
        self.data = np.load(os.path.join(data_path, f"test_data_{data_name}.npy"))
        self.v = np.load(os.path.join(data_path, f"test_v_{data_name}.npy"))
        self.label = np.load(os.path.join(data_path, f"test_label_{data_name}.npy"))
        self.test_len = self.data.shape[0]
        logger.info(f"test_len: {self.test_len}")
        logger.info(f"building test dataset from {data_path}...")

    def __init__(self):
        return self.test_len

    def __getitem__(self, index):
        return (
            self.data[index, :, :-1],
            int(self.data[index, 0, -1]),
            self.v[index],
            self.label[index],
        )


class WeightedSampler(Sampler):
    def __init__(self, data_path, data_name, replacement=True):
        # Loads a numpy array of weights from disk
        v = np.load(os.path.join(data_path, f"train_v_{data_name}.npy"))
        # Computes normalized, absolute weights from the first column
        # of this array
        self.weights = torch.as_tensor(
            np.abs(v[:, 0]) / np.sum(np.abs(v[:, 0])), dtype=torch.double
        )
        logger.info(f"weights: {self.weights}")
        self.num_samples = self.weights.shape[0]
        logger.info(f"num samples: {self.num_samples}")
        self.replacement = replacement

    def __iter__(self):
        # When iterated, uses torch.multinomial to randomly
        # sample indices from the dataset, where the probability
        # of each index being chosen is proportional to its weight
        return iter(
            torch.multinomial(
                self.weights, self.num_samples, self.replacement
            ).tolist()
        )

    def __len__(self):
        return self.num_samples
