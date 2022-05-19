from c3.c3objs import C3obj, hjson_encode
from c3.c3objs import Quantity as Qty
from c3.libraries.envelopes import envelopes
from c3.utils.tf_utils import tf_complexify
import tensorflow as tf
import numpy as np
import types
import hjson
from typing import Callable, Union, Dict

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
    shape: Callable
        function evaluating the shape in time

    """

    def __init__(
        self,
        name: str,
        desc: str = " ",
        comment: str = " ",
        params: Dict[str, Qty] = {},
        shape: Union[Callable, str] = None,
        use_t_before=False,
    ):
        if isinstance(shape, str):
            self.shape = envelopes[shape]
        else:
            self.shape = shape
        default_params = {
            "amp": Qty(value=1.0, min_val=-1.0, max_val=+1.5, unit="V"),
            "delta": Qty(value=0.0, min_val=-5.0, max_val=+5.0, unit="V"),
            "freq_offset": Qty(value=0.0, min_val=-1.0, max_val=+1.0, unit="Hz 2pi"),
            "xy_angle": Qty(value=0.0, min_val=-1.0, max_val=+1.0, unit="rad"),
            "sigma": Qty(value=5e-9, min_val=-2.0, max_val=+2.0, unit="s"),
            "t_final": Qty(value=1.0, min_val=-1.0, max_val=+1.0, unit="s"),
        }
        default_params.update(params)
        self.set_use_t_before(use_t_before)
        super().__init__(
            name=name,
            desc=desc,
            comment=comment,
            params=default_params,
        )

    def write_config(self, filepath: str) -> None:
        """
        Write dictionary to a HJSON file.
        """
        with open(filepath, "w") as cfg_file:
            hjson.dump(self.asdict(), cfg_file, default=hjson_encode)

    def asdict(self) -> dict:
        params = {}
        for key, item in self.params.items():
            params[key] = item.asdict()
        return {
            "name": self.name,
            "c3type": self.__class__.__name__,
            "shape": self.shape.__name__,
            "params": params,
        }

    def __str__(self) -> str:
        return hjson.dumps(self.asdict(), default=hjson_encode)

    def __repr__(self) -> str:
        repr_str = self.name + ":: "
        for key, item in self.params.items():
            repr_str += str(key) + " : " + str(item) + ", "
        repr_str += "shape: " + self.shape.__name__ + ", "
        return repr_str

    def set_use_t_before(self, use_t_before):
        if use_t_before:
            self.get_shape_values = self._get_shape_values_before
        else:
            self.get_shape_values = self._get_shape_values_just

    def compute_mask(self, ts, t_end) -> tf.Tensor:
        """Compute a mask to cut out a signal after t_final.

        Parameters
        ----------
        ts : tf.Tensor
            Vector of time steps.

        Returns
        -------
        tf.Tensor
            [description]
        """
        t_final = tf.minimum(self.params["t_final"].get_value(), t_end)
        dt = ts[1] - ts[0]
        return tf_complexify(
            tf.sigmoid((ts / dt + 0.001) * 1e6)
            * tf.sigmoid((0.999 * t_final - ts) / dt * 1e6)
        )

    def _get_shape_values_before(self, ts, t_final=1):
        """Return the value of the shape function at the specified times. With the offset, we make sure the
        signal starts with amplitude zero by subtracting the shape value at time -dt.

        Parameters
        ----------
        ts : tf.Tensor
            Vector of time samples.
        """
        t_before = 2 * ts[0] - ts[1]  # t[0] - (t[1] - t[0])
        offset = self.shape(t_before, self.params)
        mask = self.compute_mask(ts, t_final)
        return mask * (self.shape(ts, self.params) - offset)

    def _get_shape_values_just(self, ts, t_final=1):
        """Return the value of the shape function at the specified times.

        Parameters
        ----------
        ts : tf.Tensor
            Vector of time samples.
        """
        mask = self.compute_mask(ts, t_final)
        return mask * self.shape(ts, self.params)


@comp_reg_deco
class EnvelopeDrag(Envelope):
    def __init__(
        self,
        name: str,
        desc: str = " ",
        comment: str = " ",
        params: dict = {},
        shape: types.FunctionType = None,
        use_t_before=False,
    ):
        super().__init__(
            name=name,
            desc=desc,
            comment=comment,
            params=params,
            shape=shape,
            use_t_before=use_t_before,
        )
        self.set_use_t_before(use_t_before)

    def set_use_t_before(self, use_t_before):
        if use_t_before:
            self.base_env = super()._get_shape_values_before
        else:
            self.base_env = super()._get_shape_values_just

    def get_shape_values(self, ts, t_final=1):
        dt = ts[1] - ts[0]
        with tf.GradientTape() as t:
            t.watch(ts)
            env = tf.math.real(self.base_env(ts, t_final))
        denv = (
            t.gradient(env, ts, unconnected_gradients=tf.UnconnectedGradients.ZERO) * dt
        )  # Derivative W.R.T. to bins
        delta = self.params["delta"].get_value()
        return tf.complex(env, -denv * delta)


@comp_reg_deco
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
        use_t_before=False,
    ):
        super().__init__(
            name=name,
            desc=desc,
            comment=comment,
            params=params,
            shape=shape,
            use_t_before=use_t_before,
        )
        self.set_use_t_before(use_t_before)

    def set_use_t_before(self, use_t_before):
        if use_t_before:
            self.base_env = super()._get_shape_values_before
        else:
            self.base_env = super()._get_shape_values_just

    def get_shape_values(self, ts):
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
        shape_values = self.base_env(ts=ts_red)
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
            hjson.dump(self.asdict(), cfg_file, default=hjson_encode)

    def __repr__(self) -> str:
        repr_str = self.name + ":: "
        for key, item in self.params.items():
            repr_str += str(key) + " : " + str(item) + ", "
        return repr_str
