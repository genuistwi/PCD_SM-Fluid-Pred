import warnings
import pickle
import torch
import torch.distributed as dist
import numpy as np


def match_dim(tensor_in, target):
    """
    Adjusts the dimensions of `tensor_in` to match the number of dimensions in `target`
    by unsqueezing at the last axis until the shapes are compatible.
    """
    return tensor_in[(...,) + (None,) * (len(target.shape) - len(tensor_in.shape))]


def denormalize(data, means, stds):
    means = np.asarray(means).reshape(1, -1, *([1] * (data.ndim - 2)))
    stds = np.asarray(stds).reshape(1, -1, *([1] * (data.ndim - 2)))
    return data * stds + means


def warnings_ignore():

    # --- IMPORTANCE - None ---

    # Default weights loader method:
    warnings.filterwarnings(
        "ignore",
        message="are using `torch.load` with `weights_only=False`"
    )

    # Infos about lightning loggers, we don't care.
    warnings.filterwarnings(
        "ignore",
        message=r"Starting from v1.9.0, `tensorboardX` has been removed as a dependency.*",
        category=UserWarning,
        module=r"pytorch_lightning.*"
    )

    # --- IMPORTANCE - Moderate ---

    # Grad construction:
    """ 
    torch/autograd/graph.py:825: UserWarning: Grad strides do not match bucket view strides. 
    This may indicate grad was not created according to the gradient layout contract,  or that the param's strides 
    changed since DDP was constructed. This is not an error, but may impair performance.
    """
    warnings.filterwarnings(
        "ignore",
        message=r".*Grad strides do not match bucket view strides.*",
        category=UserWarning,
        module=r"torch\.autograd\.graph"
    )