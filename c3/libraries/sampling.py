"""Functions to select samples from a dataset by various criteria."""

import random
import numpy as np

sampling = dict()


def sampling_reg_deco(func):
    """
    Decorator for making registry of functions
    """
    sampling[str(func.__name__)] = func
    return func


@sampling_reg_deco
def all(learn_from,  batch_size):
    """
    Return all points.

    Parameters
    ----------
    learn_from : list
        List of data points
    batch_size : int
        Number of points to select

    Returns
    -------
    list
        All indeces.

    """
    total_size = len(learn_from)
    all = list(range(total_size))
    return all


@sampling_reg_deco
def from_start(learn_from,  batch_size):
    """
    Select from the beginning.

    Parameters
    ----------
    learn_from : list
        List of data points
    batch_size : int
        Number of points to select

    Returns
    -------
    list
        Selected indices.

    """
    total_size = len(learn_from)
    all = list(range(total_size))
    return all[:batch_size]


@sampling_reg_deco
def from_end(learn_from,  batch_size):
    """
    Select from the end.

    Parameters
    ----------
    learn_from : list
        List of data points
    batch_size : int
        Number of points to select

    Returns
    -------
    list
        Selected indices.

    """
    total_size = len(learn_from)
    all = list(range(total_size))
    return all[-batch_size:]


@sampling_reg_deco
def even(learn_from,  batch_size):
    """
    Select evenly distanced samples across the set.

    Parameters
    ----------
    learn_from : list
        List of data points
    batch_size : int
        Number of points to select

    Returns
    -------
    list
        Selected indices.

    """
    total_size = len(learn_from)
    all = list(range(total_size))
    n = int(np.ceil(total_size / batch_size))
    return all[::n]


@sampling_reg_deco
def random_sample(learn_from,  batch_size):
    """
    Select randomly.

    Parameters
    ----------
    learn_from : list
        List of data points.
    batch_size : int
        Number of points to select.

    Returns
    -------
    list
        Selected indices.

    """
    total_size = len(learn_from)
    all = list(range(total_size))
    return random.sample(all, batch_size)


@sampling_reg_deco
def high_std(learn_from,  batch_size):
    """
    Select points that have a high ratio of standard deviation to mean. Sampling from ORBIT data, points with a high
    std have the most coherent error, thus might be suitable for model learning. This has yet to be confirmed beyond
    anecdotal observation.

    Parameters
    ----------
    learn_from : list
        List of data points.
    batch_size : int
        Number of points to select.

    Returns
    -------
    list
        Selected indices.

    """
    a_val = []
    for sample in learn_from:
        a_val.append(np.std(sample['results'])/np.mean(sample['results']))
    return np.argsort(np.array(a_val))[-batch_size:]


@sampling_reg_deco
def even_fid(learn_from,  batch_size):
    """
    Select evenly among reached fidelities.

    Parameters
    ----------
    learn_from : list
        List of data points.
    batch_size : int
        Number of points to select.

    Returns
    -------
    list
        Selected indices.

    """
    total_size = len(learn_from)
    res = []
    for sample in learn_from:
        res.append(np.mean(sample['results']))
    n = int(np.ceil(total_size / batch_size))
    return np.argsort(np.array(res))[::n]
