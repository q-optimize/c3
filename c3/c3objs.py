"""Basic custom objects."""

import hjson
import numpy as np
import tensorflow as tf
from c3.utils.utils import num3str
from tensorflow.python.framework import ops
import copy


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
            min_val = np.min(minmax)
            max_val = np.max(minmax)
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

        self.value = None
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
        out_val = copy.deepcopy(self)
        out_val.set_value(self.get_value() + other)
        return out_val

    def __radd__(self, other):
        out_val = copy.deepcopy(self)
        out_val.set_value(self.get_value() + other)
        return out_val

    def __sub__(self, other):
        out_val = copy.deepcopy(self)
        out_val.set_value(self.get_value() - other)
        return out_val

    def __rsub__(self, other):
        out_val = copy.deepcopy(self)
        out_val.set_value(self.get_value() - other)
        return out_val

    def __mul__(self, other):
        out_val = copy.deepcopy(self)
        out_val.set_value(self.get_value() * other)
        return out_val

    def __rmul__(self, other):
        out_val = copy.deepcopy(self)
        out_val.set_value(self.get_value() * other)
        return out_val

    def __pow__(self, other):
        out_val = copy.deepcopy(self)
        out_val.set_value(self.get_value() ** other)
        return out_val

    def __rpow__(self, other):
        out_val = copy.deepcopy(self)
        out_val.set_value(other ** self.get_value())
        return out_val

    def __truediv__(self, other):
        out_val = copy.deepcopy(self)
        out_val.set_value(self.get_value() / other)
        return out_val

    def __rtruediv__(self, other):
        out_val = copy.deepcopy(self)
        out_val.set_value(other / self.get_value())
        return out_val

    def __mod__(self, other):
        out_val = copy.deepcopy(self)
        out_val.set_value(self.get_value() % other)
        return out_val

    def __lt__(self, other):
        return self.get_value() < other

    def __le__(self, other):
        return self.get_value() <= other

    def __eq__(self, other):
        return self.get_value() == other

    def __ne__(self, other):
        return self.get_value() != other

    def __ge__(self, other):
        return self.get_value() >= other

    def __gt__(self, other):
        return self.get_value() > other

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

    def subtract(self, val):
        self.set_value(self.get_value() - val)

    def add(self, val):
        self.set_value(self.get_value() + val)

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

    def set_value(self, val, extend_bounds=False) -> None:
        """Set the value of this quantity as tensorflow. Value needs to be
        within specified min and max."""
        # setting can be numpyish
        if isinstance(val, ops.EagerTensor) or isinstance(val, ops.Tensor):
            val = tf.cast(val, tf.float64)
        else:
            val = tf.constant(val, tf.float64)

        tmp = 2 * (val * self.pref - self.offset) / self.scale - 1

        if extend_bounds and tf.math.abs(tmp) > 1:
            min_val, max_val = self.get_limits()
            min_val = tf.math.reduce_min([val, min_val])
            max_val = tf.math.reduce_max([val, max_val])
            self.set_limits(min_val, max_val)
            tmp = 2 * (val * self.pref - self.offset) / self.scale - 1

        tf.debugging.assert_less_equal(
            tf.math.abs(tmp),
            tf.constant(1.0, tf.float64),
            f"Value {val.numpy()}{self.unit} out of bounds for quantity with "
            f"min_val: {num3str(self.get_limits()[0])}{self.unit} and "
            f"max_val: {num3str(self.get_limits()[1])}{self.unit}",
        )

        self.value = tf.cast(tmp, tf.float64)

        if isinstance(val, ops.EagerTensor) or isinstance(val, ops.Tensor):
            self.value = tf.cast(tmp, tf.float64)
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

    def get_limits(self):
        min_val = self.offset / self.pref
        max_val = (self.scale + self.offset) / self.pref
        return min_val, max_val

    def set_limits(self, min_val, max_val):
        val = self.get_value()
        self.offset = np.array(min_val) * self.pref
        self.scale = np.abs(np.array(max_val) - np.array(min_val)) * self.pref
        self.set_value(val)
