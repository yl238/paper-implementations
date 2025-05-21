import logging
import numpy as np

import torch

def evaluate(model, loss_fn, test_loader, params, plot_num, sample=True):
    """Evaluate the model on the test set.

    Args:
        model (_type_): _description_
        loss_fn (_type_): _description_
        test_loader (_type_): _description_
        params (_type_): _description_
        plot_num (_type_): _description_
        sample (bool, optional): _description_. Defaults to True.
    """
    model.eval()
    with torch.no_grad():
        plot_batch = np.random.randint(len(test_loader) - 1)

        summary_metric = {}
