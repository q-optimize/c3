import time
import os
import numpy as np


def log_setup(data_path, run_name=None):
    # TODO make this plattform agnostic, i.e. work with Windows(tm)
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
    ret = []
    if not hasattr(val, "__iter__"):
        val = np.array([val])
    for idx in range(val.shape[0]):
        v = val[idx]
        sign = 1
        if v == 0:
            return "0"
        if v < 0:
            v = -v
            sign = -1
        tmp = np.log10(v)
        idx = int(tmp // 3)
        if tmp < 0:
            prefix = small_units[-(idx+1)]
        else:
            prefix = big_units[idx]

        ret.append(f"{sign * (10 ** (tmp % 3)):.3f}" + prefix)
    return ret
