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
    total_size = len(learn_from)
    all = list(range(total_size))
    return all

@sampling_reg_deco
def from_start(learn_from,  batch_size):
    total_size = len(learn_from)
    all = list(range(total_size))
    return all[:batch_size]

@sampling_reg_deco
def from_end(learn_from,  batch_size):
    total_size = len(learn_from)
    all = list(range(total_size))
    return all[-batch_size:]

@sampling_reg_deco
def even(learn_from,  batch_size):
    total_size = len(learn_from)
    all = list(range(total_size))
    n = int(np.ceil(total_size / batch_size))
    return all[::n]

@sampling_reg_deco
def random_sample(learn_from,  batch_size):
    total_size = len(learn_from)
    all = list(range(total_size))
    return random.sample(all, batch_size)

@sampling_reg_deco
def high_std(learn_from,  batch_size):
    a_val = []
    for sample in learn_from:
        a_val.append(np.std(sample['results'])/np.mean(sample['results']))
    return np.argsort(np.array(a_val))[-batch_size:]

@sampling_reg_deco
def even_fid(learn_from,  batch_size):
    total_size = len(learn_from)
    res = []
    for sample in learn_from:
        res.append(np.mean(sample['results']))
    n = int(np.ceil(total_size / batch_size))
    return np.argsort(np.array(res))[::n]
