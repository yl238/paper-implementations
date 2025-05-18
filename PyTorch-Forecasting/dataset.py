import numpy as np
import torch


def generate_train_and_test(data, train_test_split=0.8):
    """
    Splits a time series array into training and testing sets.

    Args:
        data (np.array): Input time series data.
        train_test_split (float, optional): Proportion of
            data to use for training (default: 0.8).

    Returns:
        tuple: (train_data, test_data) as numpy arrays,
            each reshaped to (-1, 1).
    """
    train_size = int(len(data) * train_test_split)

    train_data = data[:train_size].reshape(-1, 1)
    test_data = data[train_size:].reshape(-1, 1)

    return train_data, test_data


def create_sequential_dataset(dataset, sequence_length=50):
    """Transform a time series into a prediction dataset. We predict the value
    immediately after a given sequence.

    Args:
        dataset (np.array): Numpy array of time series, the first dimension is time.
        sequence_length (int, optional): Size of window to look back for training.
            Defaults to 50.

    Returns:
        torch.Tensor: training dataset
    """
    features, targets = [], []

    for i in range(len(dataset) - sequence_length):
        features.append(dataset[i : i + sequence_length])
        targets.append(dataset[i + sequence_length])  # Predicting the value
        # immediately after the sequence
    features, targets = np.array(features), np.array(targets)

    # Convert data to PyTorch tensors
    features = torch.tensor(features, dtype=torch.float32)
    targets = torch.tensor(targets, dtype=torch.float32)
    return features, targets
