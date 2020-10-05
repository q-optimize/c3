from c3.c3objs import C3obj
from c3.c3objs import Quantity as Qty
import tensorflow as tf
import types


class InstructionComponent(C3obj):
    """
    Represents the components making up a pulse.

    Parameters
    ----------
    params: dict
        dictionary of the parameters needed for the shape-function to
        create the desired pulse

    """

    def __init__(
            self,
            name: str,
            desc: str = " ",
            comment: str = " ",
            params: dict = {},
            ):
        super().__init__(
            name=name,
            desc=desc,
            comment=comment
            )
        self.params = params


class Envelope(InstructionComponent):
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
        super().__init__(
            name=name,
            desc=desc,
            comment=comment,
            params=params,
            )
        self.shape = shape
        if 'amp' not in params:
            params['amp'] = Qty(
                value=0.0,
                min=-1.0,
                max=+1.0,
                unit="V"
            )
        if 'delta' not in params:
            params['delta'] = Qty(
                value=0.0,
                min=-1.0,
                max=+1.0,
                unit="V"
            )
        if 'freq_offset' not in params:
            params['freq_offset'] = Qty(
                value=0.0,
                min=-1.0,
                max=+1.0,
                unit='Hz 2pi'
            )
        if 'xy_angle' not in params:
            params['xy_angle'] = Qty(
                value=0.0,
                min=-1.0,
                max=+1.0,
                unit='rad'
            )
        if 't_final' not in params:
            params['t_final'] = Qty(
                value=0.0,
                min=-1.0,
                max=+1.0,
                unit="s"
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
        t_final = self.params['t_final']
        if t_before:
            offset = self.shape(t_before, self.params)
            vals = self.shape(ts, self.params) - offset
            mask = tf.cast(ts < t_final.numpy() - t_before, tf.float64)
        else:
            vals = self.shape(ts, self.params)
            mask = tf.cast(ts < t_final.numpy(), tf.float64)
        # With the offset, we make sure the signal starts with amplitude 0.
        return vals*mask


class Carrier(InstructionComponent):
    """Represents the carrier of a pulse."""

    def __init__(
            self,
            name: str,
            desc: str = " ",
            comment: str = " ",
            params: dict = {},
            ):
        super().__init__(
            name=name,
            desc=desc,
            comment=comment,
            params=params,
            )
