"""Miscellaneous, general utilities."""
import time
import os
import numpy as np


# SYSTEM AND SETUP
def log_setup(data_path, run_name=None):
    """
    Make sure the file path to save data exists. Create an appropriately named folder with date and time.
     Also creates a symlink "recent" to the folder.

    Parameters
    ----------
    data_path : str
        File path of where to store any data.
    run_name : str
        User specified name for the run.

    Returns
    -------
    str
        The file path to store new data.

    """
    if not os.path.isdir(data_path):
        os.makedirs(data_path)

    pwd = os.path.join(
        data_path,
        run_name,
        time.strftime("%Y_%m_%d_T_%H_%M_%S", time.localtime())
    )
    while os.path.exists(pwd):
        time.sleep(1)
        pwd = os.path.join(
            data_path,
            run_name,
            time.strftime("%Y_%m_%d_T_%H_%M_%S", time.localtime())
        )

    os.makedirs(pwd)
    recent = os.path.join(data_path, 'recent')
    replace_symlink(pwd, recent)
    return os.path.join(pwd, '')


def replace_symlink(path, alias):
    try:
        os.remove(alias)
    except FileNotFoundError:
        pass
    try:
        os.symlink(path, alias)
    except FileExistsError:
        pass

# NICE PRINTING FUNCTIONS
def eng_num(val):
    """Convert a number to engineering notation by returning number and prefix."""
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
    """Convert a number to a human readable string in engineering notation. """
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
    """Ask for y/n user decision in the command line."""
    asking = True
    text = input("(y/n): ")
    if text == 'y':
        asking = False
        boolean = True
    elif text == 'n':
        asking = False
        boolean = False
    while asking:
        text = input("Please write y or n and press enter: ")
        if text == 'y':
            asking = False
            boolean = True
        elif text == 'n':
            asking = False
            boolean = False
    return boolean
