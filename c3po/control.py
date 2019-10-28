import numpy as np
from c3po.component import C3obj


class GateSet:
    """Contains all operations and corresponding instructions."""

    def __init__(self):
        self.instructions = {}

    def add_instruction(self, instr):
        self.instructions[instr.name] = instr

    def list_parameters(self):
        par_list = []
        for gate in self.instructions.keys():
            instr = self.instructions[gate]
            for chan in instr.comps.keys():
                for comp in instr.comps[chan]:
                    for par in instr.comps[chan][comp].params:
                        par_list.append([(gate, chan, comp, par)])
        return par_list

    def get_parameters(self, opt_map: list):
        """
        Return list of values and bounds of parameters in opt_map.

        Takes a list of paramaters that are supposed to be optimized
        and returns the corresponding values and bounds.

        Parameters
        -------
        opt_map : list
            List of parameters that will be optimized, specified with a tuple
            of gate name, control name, component name and parameter.
            Parameters that are copies of each other are collected in lists.

            Example:
            opt_map = [
                [('X90p','line1','gauss1','sigma'),
                 ('Y90p','line1','gauss1','sigma')],
                [('X90p','line1','gauss2','amp')],
                [('Cnot','line1','flattop','amp')],
                [('Cnot','line2','DC','amp')]
                ]

        Returns
        -------
        opt_params : dict
            Dictionary with values, bounds lists.

            Example:
            opt_params = (
                [0,           0,           0,             0],    # Values
                [[0, 0],      [0, 0],      [0, 0],        [0.0]] # Bounds
                )

        """
        values = []
        bounds = []

        for id in opt_map:
            gate = id[0][0]
            par_id = id[0][1:4]
            gate_instr = self.instructions[gate]
            value, bound = gate_instr.get_parameter_value_bounds(par_id)
            values.append(value)
            bounds.append(bound)
        return values, bounds

    def set_parameters(self, values: list, opt_map: list):
        """Set the values in the original instruction class."""
        for indx in range(len(opt_map)):
            ids = opt_map[indx]
            for id in ids:
                gate = id[0]
                par_id = id[1:4]
                gate_instr = self.instructions[gate]
                gate_instr.set_parameter_value(par_id, values[indx])


class Instruction(C3obj):
    """
    Collection of components making up the control signal for a line.

    Parameters
    ----------
    t_start : np.float64
        Start of the signal.
    t_end : np.float64
        End of the signal.
    channels : list
        List of channel names (strings)


    Attributes
    ----------
    comps : dict
        Nested dictionary with lines and components as keys

    Example:
    comps = {
             'channel_1' : {
                            'envelope1': envelope1,
                            'envelope2': envelope2,
                            'carrier': carrier
                            }
             }
    """

    def __init__(
            self,
            name: str = " ",
            desc: str = " ",
            comment: str = " ",
            channels: list = [],
            t_start: np.float64 = 0.0,
            t_end: np.float64 = 0.0,
            ):
        super().__init__(
            name=name,
            desc=desc,
            comment=comment
            )
        self.t_start = t_start
        self.t_end = t_end
        self.comps = {}
        for chan in channels:
            self.comps[chan] = {}

    def add_component(self, comp: InstructionComponent, chan: str):
        self.comps[chan][comp.name] = comp

    def get_parameter_value_bounds(self, par_id: tuple):
        """par_id is a tuple with channel, component, parameter."""
        chan = par_id[0]
        comp = par_id[1]
        param = par_id[2]
        value = self.comps[chan][comp].params[param]
        bounds = self.comps[chan][comp].bounds[param]
        return value, bounds

    def set_parameter_value(self, par_id: tuple, value):
        """par_id is a tuple with channel, component, parameter."""
        chan = par_id[0]
        comp = par_id[1]
        param = par_id[2]
        self.comps[chan][comp].params[param] = value

    def set_parameter_bounds(self, par_id: tuple, bounds):
        chan = par_id[0]
        comp = par_id[1]
        param = par_id[2]
        self.comps[chan][comp].bounds[param] = bounds


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
            bounds: dict = {},
            ):
        super().__init__(
            name=name,
            desc=desc,
            comment=comment
            )
        self.params = params
        self.bounds = bounds
        # check that the parameters and bounds have the same key
        if params.keys() != bounds.keys():
            raise ValueError('params and bounds must have same keys')


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
            bounds: dict = {},
            shape: types.FunctionType = None,
            ):
        super().__init__(
            name=name,
            desc=desc,
            comment=comment,
            params=params,
            bounds=bounds,
            )
        self.shape = shape

    def get_shape_values(self, ts):
        """Return the value of the shape function at the specified times."""
        return self.shape(ts, self.params)


class Carrier(InstructionComponent):
    """Represents the carrier of a pulse."""

    def __init__(
            self,
            name: str,
            desc: str = " ",
            comment: str = " ",
            params: dict = {},
            bounds: dict = {},
            ):
        super().__init__(
            name=name,
            desc=desc,
            comment=comment,
            params=params,
            bounds=bounds,
            )
