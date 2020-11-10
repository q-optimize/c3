"""Basic custom objects."""

import numpy as np
import tensorflow as tf
from c3.utils.utils import num3str


class C3obj:
    """
    Represents an abstract object with parameters. To be inherited from.

    Parameters
    ----------
    name: str
        short name that will be used as identifier
    desc: str
        longer description of the component
    comment: str
        additional information about the component
    params: dict
        Parameters in this dict can be accessed and optimized
    """

    def __init__(self, name, desc="", comment="", params={}):
        self.name = name
        self.desc = desc
        self.comment = comment
        self.params = params
        for name, par in params.items():
            self.params[name] = Quantity(**par)


class Quantity:
    """
    Represents any physical quantity used in the model or the pulse
    specification. For arithmetic operations just the numeric value is used. The
    value itself is stored in an optimizer friendly way as a float between -1
    and 1. The conversion is given by
        scale (value + 1) / 2 + offset

    Parameters
    ----------
    value: np.array(np.float64) or np.float64
        value of the quantity
    min_val: np.array(np.float64) or np.float64
        minimum this quantity is allowed to take
    max_val: np.array(np.float64) or np.float64
        maximum this quantity is allowed to take
    unit: str
        physical unit
    symbol: str
        latex representation

    """

    def __init__(self, value, min_val, max_val, unit="undefined", symbol=r"\alpha"):
        if unit[-3:] == "2pi":
            pref = 2 * np.pi
        else:
            pref = 1
        self.offset = np.array(min_val * pref)
        self.scale = np.abs(np.array(max_val * pref) - np.array(min_val * pref))
        try:
            self.set_value(np.array(value * pref))
        except ValueError:
            raise ValueError(
                f"Value has to be within {min_val:.3} .. {max_val:.3}"
                f" but is {value:.3}."
            )
        self.symbol = symbol
        self.unit = unit
        if hasattr(value, "shape"):
            self.shape = value.shape
            self.length = int(np.prod(value.shape))
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
        use_prefix = True
        if self.unit == "Hz 2pi":
            val = val / 2 / np.pi
        elif self.unit == "pi":
            val = val / np.pi
            use_prefix = False
        ret = ""
        for q in num3str(val, use_prefix):
            ret += q + self.unit + " "
        return ret

    def numpy(self):
        """
        Return the value of this quantity as numpy.
        """
        return self.scale * (self.value.numpy() + 1) / 2 + self.offset

    def get_value(self, val=None):
        """
        Return the value of this quantity as tensorflow.

        Parameters
        ----------
        val : tf.float64
            Optionaly give an optimizer friendly value between -1 and 1 to
            convert to physical scale.
        """
        if val is None:
            val = self.value
        return self.scale * (val + 1) / 2 + self.offset

    def set_value(self, val):
        """ Set the value of this quantity as tensorflow. Value needs to be
        within specified min and max."""
        # setting can be numpyish
        tmp = 2 * (np.array(val) - self.offset) / self.scale - 1
        if np.any(tmp < -1) or np.any(tmp > 1):
            # TODO choose which error to raise
            # raise Exception(f"Value {val} out of bounds for quantity.")
            raise ValueError
            # TODO if we want we can extend bounds when force flag is given
        else:
            self.value = tf.constant(tmp, dtype=tf.float64)

    def get_opt_value(self):
        """ Get an optimizer friendly representation of the value."""
        return self.value.numpy().flatten()

    def set_opt_value(self, val):
        """ Set value optimizer friendly.

        Parameters
        ----------
        val : tf.float64
            Tensorflow number that will be mapped to a value between -1 and 1.
        """
        self.value = tf.acos(tf.cos(
            (tf.reshape(val, self.shape) + 1) * np.pi / 2
        )) / np.pi * 2 - 1
