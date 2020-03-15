from c3po.c3objs import C3obj
import tensorflow as tf
import types


class InstructionComponent(C3obj):
    """
    Represents the components making up a pulse.

    Parameters
    ----------
    parameters: dict
        dictionary of the parameters needed for the shape-function to
        create the desired pulse
    bounds: dict
        boundaries of the parameters, i.e. technical limits of experimental
        setup or physical boundaries. needed for optimizer

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
        # if 'amp' not in params:
        #     params['amp'] = 1.0
        # if 'freq_offset' not in params:
        #     params['freq_offset'] = 0.0
        # if 'xy_angle' not in params:
        #     params['xy_angle'] = 0.0

    def get_shape_values(self, ts, t_before=None):
        """Return the value of the shape function at the specified times."""
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
