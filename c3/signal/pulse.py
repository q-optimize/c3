from c3.c3objs import C3obj
from c3.c3objs import Quantity as Qty
from c3.libraries.envelopes import envelopes
import tensorflow as tf
import numpy as np
import types
import hjson

components = dict()


def comp_reg_deco(func):
    """
    Decorator for making registry of functions
    """
    components[str(func.__name__)] = func
    return func


@comp_reg_deco
class Envelope(C3obj):
    """
    Represents the envelopes shaping a pulse.

    Parameters
    ----------
    shape: function
        function evaluating the shape in time

    """

    def __init__(
        self,
        name: str,
        desc: str = " ",
        comment: str = " ",
        params: dict = {},
        shape: types.FunctionType = None,
    ):
        if isinstance(shape, str):
            self.shape = envelopes[shape]
        else:
            self.shape = shape
        params_default = {
            "amp": Qty(value=0.0, min_val=-1.0, max_val=+1.0, unit="V"),
            "delta": Qty(value=0.0, min_val=-5.0, max_val=+5.0, unit="V"),
            "freq_offset": Qty(value=0.0, min_val=-1.0, max_val=+1.0, unit="Hz 2pi"),
            "xy_angle": Qty(value=0.0, min_val=-1.0, max_val=+1.0, unit="rad"),
            "sigma": Qty(value=5e-9, min_val=-2.0, max_val=+2.0, unit="s"),
            "t_final": Qty(value=0.0, min_val=-1.0, max_val=+1.0, unit="s"),
        }
        params_default.update(params)
        super().__init__(
            name=name,
            desc=desc,
            comment=comment,
            params=params_default,
        )

    def write_config(self, filepath: str) -> None:
        """
        Write dictionary to a HJSON file.
        """
        with open(filepath, "w") as cfg_file:
            hjson.dump(self.asdict(), cfg_file)

    def asdict(self) -> dict:
        params = {}
        for key, item in self.params.items():
            params[key] = item.asdict()
        return {
            "c3type": self.__class__.__name__,
            "shape": self.shape.__name__,
            "params": params,
        }

    def __str__(self) -> str:
        return hjson.dumps(self.asdict())

    def get_shape_values(self, ts, t_before=None):
        """Return the value of the shape function at the specified times.

        Parameters
        ----------
        ts : tf.Tensor
            Vector of time samples.
        t_before : tf.float64
            Offset the beginning of the shape by this time.
        """
        t_final = self.params["t_final"]
        if t_before:
            offset = self.shape(t_before, self.params)
            vals = self.shape(ts, self.params) - offset
            mask = tf.cast(ts < t_final.numpy() - t_before, vals.dtype)
        else:
            vals = self.shape(ts, self.params)
            mask = tf.cast(ts < t_final.numpy(), vals.dtype)
        # With the offset, we make sure the signal starts with amplitude 0.
        return vals * mask


class EnvelopeNetZero(Envelope):
    """
    Represents the envelopes shaping a pulse.

    Parameters
    ----------
    shape: function
        function evaluating the shape in time
    params: dict
        Parameters of the envelope
        Note: t_final
    """

    def __init__(
        self,
        name: str,
        desc: str = " ",
        comment: str = " ",
        params: dict = {},
        shape: types.FunctionType = None,
    ):
        super().__init__(
            name=name,
            desc=desc,
            comment=comment,
            params=params,
            shape=shape,
        )

    def get_shape_values(self, ts, t_before=None):
        """Return the value of the shape function at the specified times.

        Parameters
        ----------
        ts : tf.Tensor
            Vector of time samples.
        t_before : tf.float64
            Offset the beginning of the shape by this time.
        """
        N_red = len(ts) // 2
        ts_red = tf.split(ts, [N_red, len(ts) - N_red], 0)[0]
        shape_values = super().get_shape_values(ts=ts_red, t_before=t_before)
        netzero_shape_values = tf.concat(
            [shape_values, -shape_values, [0] * (len(ts) % 2)], axis=0
        )
        return netzero_shape_values


@comp_reg_deco
class Carrier(C3obj):
    """Represents the carrier of a pulse."""

    def __init__(
        self,
        name: str,
        desc: str = " ",
        comment: str = " ",
        params: dict = {},
    ):
        params_default = {
            "freq": Qty(value=0.0, min_val=-1.0, max_val=+1.0, unit="V"),
            "framechange": Qty(value=0.0, min_val=-np.pi, max_val=np.pi, unit="rad"),
        }
        params_default.update(params)
        super().__init__(
            name=name,
            desc=desc,
            comment=comment,
            params=params_default,
        )

    def write_config(self, filepath: str) -> None:
        """
        Write dictionary to a HJSON file.
        """
        with open(filepath, "w") as cfg_file:
            hjson.dump(self.asdict(), cfg_file)

    def asdict(self) -> dict:
        params = {}
        for key, item in self.params.items():
            params[key] = item.asdict()
        return {"c3type": self.__class__.__name__, "params": params}
