import time
import os
import numpy as np


def log_setup(data_path, run_name=None):
    # TODO Add the name to fhe folder
    if not os.path.isdir(data_path):
        os.makedirs(data_path)
    pwd = data_path + time.strftime(
        "%Y_%m_%d_T_%H_%M_%S", time.localtime()
    )
    os.makedirs(pwd)
    recent = data_path + 'recent'
    replace_symlink(pwd, recent)
    if run_name is not None:
        name = data_path + run_name
        replace_symlink(pwd, name)
    return pwd + '/'


def replace_symlink(path, alias):
    if os.path.isdir(alias):
        os.remove(alias)
    os.symlink(path, alias)


def num3str(val):
    big_units = ['', 'K', 'M', 'G', 'T', 'P']
    small_units = ['m', 'mu', 'n', 'p', 'f']
    sign = 1
    if val < 0:
        val = -val
        sign = -1
    tmp = np.log10(val)
    idx = int(tmp // 3)
    if tmp < 0:
        prefix = small_units[idx]
    else:
        prefix = big_units[idx]

    return f"{sign * (10 ** (tmp % 3)):.3f}" + prefix
