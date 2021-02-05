"""Basic custom objects."""

import hjson
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
        pref = 1.0
        if "pi" in unit:
            pref = np.pi
        if "2pi" in unit:
            pref = 2 * np.pi

        self.pref = pref
        if min_val is None and max_val is None:
            if value != 0:
                minmax = [0.9 * value, 1.1 * value]
                min_val = min(minmax)
                max_val = max(minmax)
            else:
                min_val = -1
                max_val = 1
        self.offset = np.array(min_val) * pref
        self.scale = np.abs(np.array(max_val) - np.array(min_val)) * pref
        self.unit = unit
        self.symbol = symbol
        if hasattr(value, "shape"):
            self.shape = value.shape
            self.length = int(np.prod(value.shape))
        else:
            self.shape = ()
            self.length = 1

        self.set_value(np.array(value))

    def asdict(self):
        """
        Return a config-compatible dictionary representation.
        """
        pref = self.pref
        return {
            "value": self.numpy(),
            "min_val": self.offset / pref,
            "max_val": (self.scale / pref + self.offset / pref),
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

    def __str__(self):
        val = self.numpy()
        ret = ""
        for entry in np.nditer(val):
            ret += num3str(entry) + self.unit + " "
        return ret

    def numpy(self) -> np.ndarray:
        """
        Return the value of this quantity as numpy.
        """
        return self.get_value().numpy() / self.pref

    def get_value(self, val=None) -> tf.Tensor:
        """
        Return the value of this quantity as tensorflow.

        Parameters
        ----------
        val : tf.float64
        """
        if val is None:
            val = self.value
        return self.scale * (val + 1) / 2 + self.offset

    def set_value(self, val) -> None:
        """Set the value of this quantity as tensorflow. Value needs to be
        within specified min and max."""
        # setting can be numpyish
        tmp = 2 * (np.array(val) * self.pref - self.offset) / self.scale - 1
        if np.any(tmp < -1) or np.any(tmp > 1):
            raise Exception(
                f"Value {np.array(val)}{self.unit} out of bounds for quantity with "
                f"min_val: {num3str(self.offset / self.pref)}{self.unit} and "
                f"max_val: {num3str((self.scale + self.offset) / self.pref)}{self.unit}"
            )
            # TODO if we want we can extend bounds when force flag is given
        else:
            self.value = tf.Variable(tmp, dtype=tf.float64)

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
