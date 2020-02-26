"""C3obj and Quantity."""

import numpy as np
import tensorflow as tf
from c3po.utils.utils import num3str


class C3obj:
    """
    Represents an abstract object.

    Parameters
    ----------
    name: str
        short name that will be used as identifier
    desc: str
        longer description of the component
    comment: str
        additional information about the component

    """

    def __init__(
            self,
            name: str,
            desc: str = " ",
            comment: str = " "
            ):
        self.name = name
        self.desc = desc
        self.comment = comment
        self.params = {}

    def list_parameters(self):
        par_list = []
        for par_key in sorted(self.params.keys()):
            par_id = (self.name, par_key)
            par_list.append(par_id)
        return par_list

    def print_parameter(self, par_id):
        print(self.params[par_id])


class Quantity:
    """
    Represents any parameter used in the model or the pulse speficiation. For
    arithmetic operations just the numeric value is used.

    Parameters
    ----------
    value: np.array(np.float64) or np.float64
        value of the quantity
    min: np.array(np.float64) or np.float64
        minimun params this quantity is allowed to take
    max: np.array(np.float64) or np.float64
        maximum params this quantity is allowed to take
    symbol: str
        latex representation
    unit: str
        physical unit

    """

    def __init__(
        self,
        # TODO how to specify two options for type
        value,
        min,
        max,
        symbol: str = '\\alpha',
        unit: str = 'a.u.'
    ):
        self.offset = np.array(min)
        self.scale = np.abs(np.array(max) - np.array(min))
        # TODO if setting is out of bounds this double breaks
        self.set_value(value)
        self.symbol = symbol
        self.unit = unit
        if hasattr(value, "shape"):
            self.shape = value.shape
            self.length = np.prod(value.shape)
        else:
            self.shape = ()
            self.length = 1

    def __add__(self, other):
        return self.numpy() + other

    def __radd__(self, other):
        return self.numpy() + other

    def __sub__(self, other):
        return self.numpy() - other

    def __rsub__(self, other):
        return other - self.numpy()

    def __mul__(self, other):
        return self.numpy() * other

    def __rmul__(self, other):
        return self.numpy() * other

    def __pow__(self, other):
        return self.numpy() ** other

    def __rpow__(self, other):
        return other ** self.numpy()

    def __truediv__(self, other):
        return self.numpy() / other

    def __rtruediv__(self, other):
        return other / self.numpy()

    def __str__(self):
        val = self.numpy()
        if self.unit == "Hz 2pi":
            val = val / 2 / np.pi
        ret = ""
        for q in num3str(val):
            ret += q + self.unit + " "
        return ret

    def numpy(self):
        return self.scale * np.arcsin(self.value.numpy()) * 2 / np.pi + self.offset

    def get_value(self, val=None):
        if val is None:
            val = self.value
        return self.scale * (tf.asin(val) * 2 / np.pi) + self.offset

    def set_value(self, val):
        # setting can be numpyish
        tmp = (np.array(val) - self.offset) / self.scale
        if np.any(tmp < -1) or np.any(tmp > 1):
            # TODO choose which error to raise
            # raise Exception(f"Value {val} out of bounds for quantity.")
            raise ValueError()
            # TODO if we want we can extend bounds when force flag is given
        else:
            self.value = tf.sin(
                (tf.constant(tmp, dtype=tf.float64)) * np.pi / 2
            )

    def get_opt_value(self):
        return np.arcsin(self.value.numpy().flatten()) * 2 / np.pi

    def set_opt_value(self, val):
        self.value = tf.sin(tf.reshape(val, self.shape) * np.pi / 2)
