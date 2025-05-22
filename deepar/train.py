import argparse
import logging
import os

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from config import config
from model.net import loss_fn
from torch.utils.data import DataLoader
from torch.utils.data.sampler import RandomSampler
from tqdm import tqdm

import params

logger = logging.getLogger(__name__)


def train(model: nn.Module, optimizer: optim, loss_fn, train_loader: DataLoader, test_loader: DataLoader,
          params: params.Params,
          epoch: int) -> float:
    """Train the model on one epoch by batches

    Args:
        model (nn.Module): _description_
        optimizer (optim): _description_
        loss_fn (_type_): _description_
        train_loader (DataLoader): _description_
        test_loader (DataLoader): _description_
        params (params.Params): _description_
        epoch (int): _description_

    Returns:
        float: _description_
    """
    model.train()
    loss_epoch = np.zeros(len(train_loader))

    for i, (train_batch, idx, labels_batch) in enumerate(tqdm(train_loader)):
        optimizer.zero_grad()
        batch_size = train_batch.shape[0]

        train_batch = train_batch.permute(1, 0, 2).to(torch.float32) # Not scaled
        labels_batch = labels_batch.permute(1, 0).to(torch.float32) # Not scaled
        idx = idx.unsqueeze(0)

        loss = torch.zeros(1)
        hidden = model.init_hidden(batch_size)
        cell = model.init_cell(batch_size)

        for t in range(params.train_window):
            # if z_t is missing, replace it by output mu from the last time step
            zero_index = (train_batch[t, :, 0] == 0)
            mu, sigma, hidden, cell = model(train_batch[t].unsqueeze_(0).clone(), idx, hidden, cell)
            loss += loss_fn(mu, sigma, labels_batch[t])

        loss.backward()
        optimizer.step()
        loss = loss.item() / params.train_window # loss per time step
        loss_epoch[i] = loss
        


def train_and_evaluate(
    model: nn.Module,
    train_loader: DataLoader,
    test_loader: DataLoader,
    optimizer: optim,
    loss_fn,
    params,
    restore_file: str=None) -> None:
    """Train the model and evaluate every epoch.

    Args:
        model (nn.Module): The DeepAR model
        train_loader (DataLoader): load train data and labels
        test_loader (DataLoader): load test data and labels
        optimizer (optim): Optimizer for parameters of model
        loss_fn: A function that takes outputs and labels per 
            timestep, and then computes the loss for the batch
        params (Params): hyperparameters
        restore_file (str, optional): Name of the file to restore
            from (without its extention .pth.tar). Defaults to None.
    """
    # Reload weights from restore_file if specified
    if restore_file is not None:
        restore_path = os.path.join(params.model_dir, restore_file + '.pth.tar')
        logger.info(f'Restoring parameters from {restore_path}')
    
    logger.info('Begin training and evaluation')


if __name__ == "__main__":
    data_dir = os.path.join("data", config["local_path"])
    file = os.path.join(data_dir, "train_data_elect.npy")

    data = np.load(file)
    print(data.shape)
    print(data[:20, :20])
