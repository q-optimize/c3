import time
import os
import numpy as np


def log_setup(data_path):
    if not os.path.isdir(data_path):
        os.makedirs(data_path)
    pwd = data_path + time.strftime(
        "%Y_%m_%d_T_%H_%M_%S", time.localtime()
    )
    os.makedirs(pwd)
    recent = data_path + 'recent'
    if os.path.isdir(recent):
        os.remove(recent)
    os.symlink(pwd, recent)
    return pwd + '/'


def num3str(val):
    big_units = ['', 'K', 'M', 'G', 'T', 'P']
    small_units = ['m', 'mu', 'n', 'p', 'f']
    tmp = np.log10(val)
    idx = int(tmp // 3)
    if tmp < 0:
        prefix = small_units[idx]
    else:
        prefix = big_units[idx]

    return f"{10 ** (tmp % 3):.3f}" + prefix
