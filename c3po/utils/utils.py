import time
import os
import numpy as np


# SYSTEM AND SETUP
def log_setup(data_path, run_name=None):
    # TODO make this plattform agnostic, i.e. work with Windows(tm)
    # TODO Add the name to fhe folder
    if not os.path.isdir(data_path):
        os.makedirs(data_path)
    pwd = data_path + "/" +time.strftime(
        "%Y_%m_%d_T_%H_%M_%S", time.localtime()
    )
    try:
        os.makedirs(pwd)
    except FileExistsError:
        pass
    recent = data_path + 'recent'
    replace_symlink(pwd, recent)
    if run_name is not None:
        name = data_path + run_name
        replace_symlink(pwd, name)
    return pwd + '/'


def replace_symlink(path, alias):
    try:
        os.remove(alias)
    except FileNotFoundError:
        pass
    os.symlink(path, alias)


# NICE PRINTNG FUNCTIONS
def eng_num(val):
    big_units = ['', 'K', 'M', 'G', 'T', 'P', 'E', 'Z']
    small_units = ['m', 'µ', 'n', 'p', 'f', 'a', 'z']
    sign = 1
    if val == 0:
        return 0, ""
    if val < 0:
        val = -val
        sign = -1
    tmp = np.log10(val)
    idx = int(tmp // 3)
    if tmp < 0:
        prefix = small_units[-(idx+1)]
    else:
        prefix = big_units[idx]

    return sign * (10 ** (tmp % 3)), prefix


def num3str(val, use_prefix=True):
    ret = []
    if not hasattr(val, "__iter__"):
        val = np.array([val])
    for idx in range(val.shape[0]):
        v = val[idx]
        if use_prefix:
            num, prefix = eng_num(v)
            ret.append(f"{num:.3f} " + prefix)
        else:
            ret.append(f"{v:.3f} ")
    return ret


# USER INTERACTION
def ask_yn():
    asking = True
    text = input("(y/n): ")
    if text == 'y' or text == 'HELL YEAH':
        asking = False
        boolean = True
    elif text == 'n' or text == 'FUCK NO':
        asking = False
        boolean = False
    while asking:
        text = input("Please write y or n and press enter: ")
        if text == 'y' or text == 'HELL YEAH':
            asking = False
            boolean = True
        elif text == 'n' or text == 'FUCK NO':
            asking = False
            boolean = False
    return boolean
