"""Miscellaneous, general utilities."""
import time
import os
import numpy as np
from tensorflow.python.framework import ops
from typing import Tuple
import warnings


# SYSTEM AND SETUP
def log_setup(data_path: str, run_name: str = "run") -> str:
    """
    Make sure the file path to save data exists. Create an appropriately named
    folder with date and time. Also creates a symlink "recent" to the folder.

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
        data_path, run_name, time.strftime("%Y_%m_%d_T_%H_%M_%S", time.localtime())
    )
    while os.path.exists(pwd):
        time.sleep(1)
        pwd = os.path.join(
            data_path, run_name, time.strftime("%Y_%m_%d_T_%H_%M_%S", time.localtime())
        )

    os.makedirs(pwd)
    recent = os.path.join(data_path, "recent")
    replace_symlink(pwd, recent)
    return os.path.join(pwd, "")


def replace_symlink(path: str, alias: str) -> None:
    """Create a symbolic link."""
    try:
        os.remove(alias)
    except FileNotFoundError:
        pass
    except PermissionError:
        warnings.warn("Could not remove symlink")
    try:
        os.symlink(path, alias)
    except FileExistsError:
        pass
    except OSError:
        warnings.warn("OSError encountered while creating symlink")
    except PermissionError:
        warnings.warn("Permission encountered while creating symlink")


# NICE PRINTING FUNCTIONS
def eng_num(val: float) -> Tuple[float, str]:
    """Convert a number to engineering notation by returning number and prefix."""
    if np.array(val).size > 1:
        return np.array(val), ""
    if np.isnan(val):
        return np.nan, "NaN"
    big_units = ["", "K", "M", "G", "T", "P", "E", "Z"]
    small_units = ["m", "µ", "n", "p", "f", "a", "z"]
    sign = 1
    if val == 0:
        return 0, ""
    if val < 0:
        val = -val
        sign = -1
    tmp = np.log10(val)
    idx = int(tmp // 3)
    if tmp < 0:
        if np.abs(idx) > len(small_units):
            return val * (10 ** (3 * len(small_units))), small_units[-1]
        prefix = small_units[-(idx + 1)]
    else:
        if np.abs(idx) > len(big_units) - 1:
            return val * (10 ** (-3 * (len(big_units) - 1))), big_units[-1]
        prefix = big_units[idx]

    return sign * (10 ** (tmp % 3)), prefix


def num3str(val: float, use_prefix: bool = True) -> str:
    """Convert a number to a human readable string in engineering notation. """
    if np.array(val).size > 1:
        return np.array2string(val, precision=3)
    if use_prefix:
        num, prefix = eng_num(val)
        formatted_string = f"{num:.3f} " + prefix
    else:
        formatted_string = f"{val:.3} "
    return formatted_string


# USER INTERACTION
def ask_yn() -> bool:
    """Ask for y/n user decision in the command line."""
    asking = True
    text = input("(y/n): ")
    if text == "y":
        asking = False
        boolean = True
    elif text == "n":
        asking = False
        boolean = False
    while asking:
        text = input("Please write y or n and press enter: ")
        if text == "y":
            asking = False
            boolean = True
        elif text == "n":
            asking = False
            boolean = False
    return boolean


def jsonify_list(data):
    if isinstance(data, dict):
        return {str(k): jsonify_list(v) for k, v in data.items()}
    elif isinstance(data, list):
        return [jsonify_list(v) for v in data]
    elif isinstance(data, tuple):
        return tuple(jsonify_list(v) for v in data)
    elif isinstance(data, np.ndarray):
        return data.tolist()
    elif isinstance(data, ops.EagerTensor):
        return data.numpy().tolist()
    else:
        return data


def deprecated(message: str):
    """Decorator for deprecating functions

    Parameters
    ----------
    message : str
        Message to display along with DeprecationWarning

    Examples
    --------
    Add a :code:`@deprecated("message")` decorator to the function::

        @deprecated("Using standard width. Better use gaussian_sigma.")
        def gaussian(t, params):
            ...

    """

    def deprecated_decorator(func):
        def deprecated_func(*args, **kwargs):
            warnings.warn(
                "{} is a deprecated function. {}".format(func.__name__, message),
                category=DeprecationWarning,
                stacklevel=2,
            )
            warnings.simplefilter("default", DeprecationWarning)
            return func(*args, **kwargs)

        return deprecated_func

    return deprecated_decorator
