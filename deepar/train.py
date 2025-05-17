import argparse
import logging
import os

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data.sampler import RandomSampler
from tqdm import tqdm

from config import config


if __name__ == '__main__':
    data_dir = os.path.join('data', config['local_path'])
    file = os.path.join(data_dir, 'train_data_elect.npy')
    

    data = np.load(file)
    print(data.shape)
    print(data[:20, :20])
