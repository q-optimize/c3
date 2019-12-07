import types
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

    def get_parameters(self, opt_map=None, scaled=False, to_str=False):
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
                )

        """
        values = []
        if opt_map is None:
            opt_map = self.list_parameters()
        for id in opt_map:
            gate = id[0][0]
            par_id = id[0][1:4]
            gate_instr = self.instructions[gate]
            if scaled:
                value = gate_instr.get_parameter_value_scaled(par_id)
            elif to_str:
                value = str(gate_instr.get_parameter_qty(par_id))
            else:
                value = gate_instr.get_parameter_value(par_id)
            values.append(value)
        return values

    def set_parameters(self, values: list, opt_map: list, scaled=False):
        """Set the values in the original instruction class."""
        # TODO catch key errors
        for indx in range(len(opt_map)):
            ids = opt_map[indx]
            for id in ids:
                gate = id[0]
                par_id = id[1:4]
                gate_instr = self.instructions[gate]
                if scaled:
                    gate_instr.set_parameter_value_scaled(
                        par_id, values[indx]
                    )
                else:
                    gate_instr.set_parameter_value(
                        par_id, values[indx]
                    )


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
        if 'amp' not in params:
            params['amp'] = 1.0
        if 'freq_offset' not in params:
            params['freq_offset'] = 0.0

    def get_shape_values(self, ts):
        """Return the value of the shape function at the specified times."""
        dt = ts[1] - ts[0]
        offset = self.shape(ts[0]-dt, self. params)
        # With the offset, we make sure the signal starts with amplitude 0.
        return self.shape(ts, self.params) - offset


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
        # TODO remove redundancy of channels in instruction

    def add_component(self, comp: InstructionComponent, chan: str):
        self.comps[chan][comp.name] = comp

    # TODO There's a number of get_parameter functions in different places
    # that do similar things. Think about this.
    # TODO Combine these get parameter functions
    def get_parameter_value_scaled(self, par_id: tuple):
        """par_id is a tuple with channel, component, parameter."""
        chan = par_id[0]
        comp = par_id[1]
        param = par_id[2]
        # The parameter can be a c3po.Quanity or a float, this cast makes it
        # work in both cases.
        value = self.comps[chan][comp].params[param].value
        return value

    def get_parameter_value(self, par_id: tuple):
        """par_id is a tuple with channel, component, parameter."""
        chan = par_id[0]
        comp = par_id[1]
        param = par_id[2]
        # The parameter can be a c3po.Quanity or a float, this cast makes it
        # work in both cases.
        value = float(self.comps[chan][comp].params[param])
        return value

    def get_parameter_qty(self, par_id: tuple):
        """par_id is a tuple with channel, component, parameter."""
        chan = par_id[0]
        comp = par_id[1]
        param = par_id[2]
        # The parameter can be a c3po.Quanity or a float, this cast makes it
        # work in both cases.
        value = self.comps[chan][comp].params[param]
        return value

    # TODO Consider putting this code directly in the gateset object
    def set_parameter_value_scaled(self, par_id: tuple, value):
        """par_id is a tuple with channel, component, parameter."""
        chan = par_id[0]
        comp = par_id[1]
        param = par_id[2]
        self.comps[chan][comp].params[param].tf_set_value(value)

    def set_parameter_value(self, par_id: tuple, value):
        """par_id is a tuple with channel, component, parameter."""
        chan = par_id[0]
        comp = par_id[1]
        param = par_id[2]
        try:
            self.comps[chan][comp].params[param].set_value(value)
        except Exception:
            print("Value out of bounds")
            print(f"Trying to set {par_id} to value {value}")
