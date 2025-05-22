import json
import logging
import os
import shutil

import numpy as np
import torch
from tqdm import tqdm

logger = logging.getLogger("DeepAR.utils")


def set_logger(log_path):
    """Set the logger to log info in terminal and file `log_path`.
    In general, it is useful to have a logger so that every output
    to the terminal is saved in a permanent file. Here we save it to
    `model_dir/train.log`.

    Args:
        log_path (string): Path of the log file.
    """
    _logger = logging.getLogger("DeepAR")
    _logger.setLevel(logging.INFO)
    fmt = logging.Formatter("[%(asctime)s] %(name)s: %(message)s", "%H:%M:%S")

    class TqdmHandler(logging.StreamHandler):
        def __init__(self, formatter):
            logging.StreamHandler.__init__(self)
            self.setFormatter(formatter)

        def emit(self, record):
            message = self.format(record)
            tqdm.write(message)

    file_handler = logging.FileHandler(log_path)
    file_handler.setFormatter(fmt)
    _logger.addHandler(file_handler)
    _logger.addHandler(TqdmHandler(fmt))


def save_dict_to_json(d, json_path):
    """Save dict of floats in json file.

    Args:
        d (dict): Dictionary of float-castable values (np.float, int, float, etc)
        json_path (string): Path to json file
    """
    with open(json_path, "w") as f:
        # We need to convert the values to float for JSON
        # (it doesn't accept np.array, np.float)
        d = {k: float(v) for k, v in d.items()}
        json.dump(d, f, indent=4)


def save_checkpoint(state, is_best, epoch, checkpoint, ins_name=-1):
    """Saves model and training parameters at checkpoint + '.last.pth.tar'.
    If `is_best==True`, also saves checkpoint + 'best.pth.tar'

    Args:
        state (dict): Model's state_dict, may contain other keys such as epoch, optimizer
        is_best (bool): True if it is the best model seen till now.
        epoch (int): Epoch count.
        checkpoint (string): Folder where parameters are to be saved.
        ins_name (int, optional): instance index. Defaults to -1.
    """
    if ins_name == -1:
        file_path = os.path.join(checkpoint, f"epoch_{epoch}.pth.tar")
    else:
        file_path = os.path.join(checkpoint, f"epoch_{epoch}_ins_{ins_name}.pth.tar")

    if not os.path.exists(checkpoint):
        logger.info(
            f"Checkpoint directory does not exist! Making directory {checkpoint}"
        )
        os.mkdir(checkpoint)

    torch.save(state, file_path)
    logger.info(f"Checkpoint saved to {file_path}")
    if is_best:
        shutil.copyfile(file_path, os.path.join(checkpoint, "best.pth.tar"))
        logger.info("Best checkpoint copied to best.pth.tar")


def load_checkpoint(checkpoint, model, optimizer=None):
    """Loads model parameters (state_dict) from file_path. If optimizer is provided,
    loads `state_dict` of optimizer assuming it is present in checkpoint.

    Args:
        checkpoint (string): Filename which needs to be loaded
        model (torch.nn.Module): Model for which the parameters are loaded
        optimizer (torch.optim, optional): Resume optimizer from checkpoint. Defaults to None.
    """
    if not os.path.exists(checkpoint):
        raise FileNotFoundError(f"File doesn't exist {checkpoint}")

    checkpoint = torch.load(checkpoint)
    model.load_state_dict(checkpoint["state_dict"])

    if optimizer:
        optimizer.load_state_dict(checkpoint["optim_dict"])
    return checkpoint


def init_metrics(sample=True):
    metrics = {
        "ND": np.zeros(2),  # numerator, denominator
        "RMSE": np.zeros(3),  # numerator, denominator, time step count
        "test_loss": np.zeros(2),
    }
    if sample:
        metrics["rou90"] = np.zeros(2)
        metrics["rou50"] = np.zeros(2)
    return metrics

