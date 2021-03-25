"""Basic custom objects."""

import hjson
import numpy as np
import tensorflow as tf
from c3.utils.utils import num3str
from tensorflow.python.framework import ops


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

    def __init__(self, name, desc="", comment="", params=None):
        self.name = name
        self.desc = desc
        self.comment = comment
        self.params = {}
        if params:
            for pname, par in params.items():
                # TODO params here should be the dict representation only
                if isinstance(par, Quantity):
                    self.params[pname] = par
                else:
                    try:
                        self.params[pname] = Quantity(**par)
                    except Exception as exception:
                        print(f"Error initializing {pname} with\n {par}")
                        raise exception

    def __str__(self) -> str:
        return hjson.dumps(self.asdict())

    def asdict(self) -> dict:
        params = {}
        for key, item in self.params.items():
            params[key] = item.asdict()
        return {"c3type": self.__class__.__name__, "params": params}


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

    def __init__(
        self, value, unit="undefined", min_val=None, max_val=None, symbol=r"\alpha"
    ):
        value = np.array(value)
        if "pi" in unit:
            pref = np.pi
        if "2pi" in unit:
            pref = 2 * np.pi
        else:
            pref = 1.0
        self.pref = pref
        if min_val is None and max_val is None:
            minmax = [0.9 * value, 1.1 * value]
            min_val = np.min(minmax, axis=0)
            max_val = np.max(minmax, axis=0)
        self.offset = np.array(min_val) * pref
        self.scale = np.abs(np.array(max_val) - np.array(min_val)) * pref
        self.unit = unit
        self.symbol = symbol
        if hasattr(value, "shape"):
            self.shape = value.shape
            self.length = int(np.prod(value.shape))
        else:
            self.shape = (1,)
            self.length = 1

        self.set_value(np.array(value))

    def asdict(self) -> dict:
        """
        Return a config-compatible dictionary representation.
        """
        pref = self.pref
        return {
            "value": self.numpy().tolist(),
            "min_val": (self.offset / pref).tolist(),
            "max_val": (self.scale / pref + self.offset / pref).tolist(),
            "unit": self.unit,
            "symbol": self.symbol,
        }

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

    def __mod__(self, other):
        return self.numpy() % other

    def __array__(self):
        return np.array(self.numpy())

    def __len__(self):
        return self.length

    def __getitem__(self, key):
        if self.length == 1 and key == 0:
            return self.numpy()
        return self.numpy().__getitem__(key)

    def __float__(self):
        if self.length > 1:
            return NotImplemented
        return float(self.numpy())

    def __repr__(self):
        return self.__str__()[:-1]

    def __str__(self):
        val = self.numpy()
        ret = ""
        for entry in np.nditer(val):
            if self.unit != "undefined":
                ret += num3str(entry) + self.unit + " "
            else:
                ret += num3str(entry, use_prefix=False) + " "
        return ret

    def numpy(self) -> np.ndarray:
        """
        Return the value of this quantity as numpy.
        """
        return self.get_value().numpy() / self.pref

    def get_value(self, val: tf.float64 = None, dtype: tf.dtypes = None) -> tf.Tensor:
        """
        Return the value of this quantity as tensorflow.

        Parameters
        ----------
        val : tf.float64
        dtype: tf.dtypes
        """
        if val is None:
            val = self.value
        if dtype is None:
            dtype = self.value.dtype

        # TODO: cleanup mashup between numpy and tensorflow (scale and offset or numpyish)
        value = self.scale * (val + 1) / 2 + self.offset
        return tf.cast(value, dtype)

    def set_value(self, val) -> None:
        """Set the value of this quantity as tensorflow. Value needs to be
        within specified min and max."""
        # setting can be numpyish
        tmp = (
            2 * (tf.cast(val, dtype=tf.float64) * self.pref - self.offset) / self.scale
            - 1
        )
        if tf.reduce_any([tmp < -1, tmp > 1]):
            raise Exception(
                f"Value {np.array(val)}{self.unit} out of bounds for quantity with "
                f"min_val: {num3str(self.offset / self.pref)}{self.unit} and "
                f"max_val: {num3str((self.scale + self.offset) / self.pref)}{self.unit}"
            )
            # TODO if we want we can extend bounds when force flag is given
        else:
            if isinstance(val, ops.EagerTensor) or isinstance(val, ops.Tensor):
                self.value = tf.cast(
                    2 * (val * self.pref - self.offset) / self.scale - 1, tf.float64
                )
            else:
                self.value = tf.constant(tmp, tf.float64)

    def get_opt_value(self) -> np.ndarray:
        """ Get an optimizer friendly representation of the value."""
        return self.value.numpy().flatten()

    def set_opt_value(self, val: float) -> None:
        """Set value optimizer friendly.

        Parameters
        ----------
        val : tf.float64
            Tensorflow number that will be mapped to a value between -1 and 1.
        """
        bound_val = tf.cos((tf.reshape(val, self.shape) + 1) * np.pi / 2)
        self.value = tf.acos(bound_val) / np.pi * 2 - 1
