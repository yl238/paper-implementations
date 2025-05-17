import os


import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tqdm
from scipy import stats
from config import config

def prep_data(
    data,
    covariates,
    data_start,
    num_series,
    total_time,
    num_covariates,
    window_size,
    stride_size,
    train=True,
):
    """Prepares windowed input, covariate and label arrays for time series
    forecasting.

    Args:
        data (np.ndarray): The main time series data, shape (time_len, num_series).
        covariates (np.ndarray): Covariate features, (time_len, num_covariates).
        data_start (np.ndarray): Array of start indices for each series.
        num_series (int): Number of time series.
        total_time (int): Total time steps in the data.
        num_covariates (int): Number of covariate features.
        window_size (int): Length of each input window.
        stride_size (int): Step size between windows
        train (bool, optional): If true, prepare training data, else test data. Defaults to True.

    Returns:
        x_input (np.ndarray): Input array for the model, shape
            (total_windows, window_size, 1 + num_covariates + 1).
        v_input (np.ndarray): Normalization factors, shape (total_windows, 2).
        label (np.ndarray): Target labels, shape (total_windows, window_size).
    """
    # Calculate the number of windows per series
    time_len = data.shape[0]
    input_size = window_size - stride_size

    windows_per_series = np.full(
        (num_series), (time_len - input_size) // stride_size
    )
    if train:
        # Adjust windows for training based on data_start
        windows_per_series -= (data_start + stride_size - 1) // stride_size
    print("data_start: ", data_start.shape)
    print(data_start)
    print("windows: ", windows_per_series)
    print(windows_per_series)

    total_windows = np.sum(windows_per_series)
    # Initialize arrays for input, label, and normalisation
    x_input = np.zeros(
        (total_windows, window_size, 1 + num_covariates + 1), dtype="float32"
    )
    label = np.zeros((total_windows, window_size), dtype="float32")
    # Computed weights used for sampling
    v_input = np.zeros((total_windows, 2), dtype="float32")
    count = 0
    if not train:
        # For test, use only the last time_len covariates
        covariates = covariates[-time_len:]
    # Iterate over each series to create windows
    for series in tqdm.trange(num_series):
        # Create a covariate "age" feature for each time step after data_start
        cov_age = stats.zscore(np.arange(total_time - data_start[series]))
        if train:
            covariates[data_start[series] : time_len, 0] = cov_age[
                : time_len - data_start[series]
            ]
        else:
            covariates[:, 0] = cov_age[-time_len:]
        for i in range(windows_per_series[series]):
            # Calculate window start and end indices
            if train:
                window_start = stride_size * i + data_start[series]
            else:
                window_start = stride_size * i
            window_end = window_start + window_size

            print("x: ", x_input[count, 1:, 0].shape)
            print("window start: ", window_start)
            print("window end: ", window_end)
            print("data: ", data.shape)
            print("d: ", data[window_start : window_end - 1, series].shape)

            # Assign data and covariates to input arrays
            x_input[count, 1:, 0] = data[window_start : window_end - 1, series]
            x_input[count, :, 1 : 1 + num_covariates] = covariates[
                window_start:window_end, :
            ]
            x_input[count, :, -1] = series  # Series index as a feature

            label[count, :] = data[window_start:window_end, series]

            # Compute normalization factor for the window
            nonzero_sum = (x_input[count, 1:input_size, 0] != 0).sum()
            if nonzero_sum == 0:
                v_input[count, 0] = 0
            else:
                v_input[count, 0] = (
                    np.true_divide(
                        x_input[count, 1:input_size, 0].sum(), nonzero_sum
                    )
                    + 1
                )
                # Normalize input and label by the computed factor
                x_input[count, :, 0] = x_input[count, :, 0] / v_input[count, 0]
                if train:
                    label[count, :] = label[count, :] / v_input[count, 0]
            count += 1
    # Save training and test data as NumPy dumps
    prefix = os.path.join(save_path, "train_" if train else "test_")
    np.save(prefix + "data_" + save_name, x_input)
    np.save(prefix + "v_" + save_name, v_input)
    np.save(prefix + "label_" + save_name, label)
    return None


def gen_covariates(times, num_covariates):
    covariates = np.zeros((times.shape[0], num_covariates))
    for i, input_time in enumerate(times):
        covariates[i, 1] = input_time.weekday()
        covariates[i, 2] = input_time.hour
        covariates[i, 3] = input_time.month
    for i in range(1, num_covariates):
        covariates[:, i] = stats.zscore(covariates[:, i])
    return covariates[:, :num_covariates]


def visualize(data, week_start, window_size):
    x = np.arange(window_size)
    f = plt.figure()
    plt.plot(x, data[week_start : week_start + window_size], color="b")
    f.savefig("visual.png")
    plt.close()


if __name__ == "__main__":
    name = config['name']
    save_name = config['local_path']

    window_size = config['window_size']
    stride_size = config['stride_size']
    num_covariates = config['num_covariates']

    train_start = config['train_start']
    train_end = config['train_end']
    test_start = config['test_start']
    test_end = config['test_end']

    pred_days = config['pred_days']
    given_days = config['given_days']

    save_path = os.path.join("data", save_name)
    csv_path = os.path.join(save_path, name)
    df = pd.read_csv(csv_path, sep=";", index_col=0, parse_dates=True, decimal=',')
    df.index = pd.to_datetime(df.index)
    df = df.resample("h", label="left", closed="right").sum()[train_start:test_end]
    
    df.fillna(0, inplace=True)
    covariates = gen_covariates(df[train_start:train_end].index, num_covariates)

    train_data = df[train_start:train_end].values
    print(len(df))

    test_data = df[test_start:test_end].values
    data_start = (train_data != 0).argmax(
        axis=0
    )  # Find first non-zero value in each time series
    total_time = df.shape[0]
    num_series = df.shape[1]

    # prepare training and test data, will save in the directory `data/elect`
    # as numpy dumps.
    prep_data(
        data=train_data,
        covariates=covariates,
        data_start=data_start,
        num_series=num_series,
        num_covariates=num_covariates,
        total_time=total_time,
        window_size=window_size,
        stride_size=stride_size,
    )
    prep_data(
        data=test_data,
        covariates=covariates,
        data_start=data_start,
        num_series=num_series,
        num_covariates=num_covariates,
        total_time=total_time,
        window_size=window_size,
        stride_size=stride_size,
        train=False
    )
