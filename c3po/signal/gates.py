import copy
import numpy as np
from c3po.signal.pulse import InstructionComponent


class GateSet:
    """Contains all operations and corresponding instructions."""

    def __init__(self):
        self.instructions = {}

    def write_config(self):
        cfg = {}
        for instr in self.instructions:
            cfg[instr] = self.instructions[instr].write_config()
        return cfg

    def add_instruction(self, instr):
        # TODO make this use ___dict___ ?
        self.instructions[instr.name] = instr
        self.update_par_lens()

    def list_parameters(self):
        par_list = []
        for gate in self.instructions.keys():
            instr = self.instructions[gate]
            for chan in instr.comps.keys():
                for comp in instr.comps[chan]:
                    for par in instr.comps[chan][comp].params:
                        par_list.append([(gate, chan, comp, par)])
        return par_list

    def update_par_lens(self):
        opt_map = self.list_parameters()
        id_list = []
        par_lens = []
        for ids in opt_map:
            for id in ids:
                id_list.append(id)
                gate = id[0]
                chan = id[1]
                comp = id[2]
                param = id[3]
                gate_instr = self.instructions[gate]
                par = gate_instr.comps[chan][comp].params[param]
                par_lens.append(par.length)
        self.par_lens = par_lens
        self.id_list = id_list

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
            chan = id[0][1]
            comp = id[0][2]
            param = id[0][3]
            gate_instr = self.instructions[gate]
            par = gate_instr.comps[chan][comp].params[param]
            if scaled:
                values.extend(par.get_opt_value())
            elif to_str:
                values.append(str(par))
            else:
                values.append(par.get_value())
        return values

    def set_parameters(self, values: list, opt_map: list, scaled=False):
        """Set the values in the original instruction class."""
        # TODO catch key errors
        val_indx = 0
        for indx in range(len(opt_map)):
            ids = opt_map[indx]
            for id in ids:
                id_indx = self.id_list.index(id)
                par_len = self.par_lens[id_indx]
                gate = id[0]
                par_id = id[1:4]
                chan = par_id[0]
                comp = par_id[1]
                param = par_id[2]
                gate_instr = self.instructions[gate]
                par = gate_instr.comps[chan][comp].params[param]
                if scaled:
                    val = values[val_indx:val_indx+par_len]
                    par.set_opt_value(val)
                else:
                    try:
                        val = values[val_indx]
                        if len(id) == 5:
                            fct = id[4]
                            val = fct(val)
                        par.set_value(val)
                    except ValueError:
                        raise ValueError(f"Trying to set {id} to value {val}")
            if scaled:
                val_indx += par_len
            else:
                val_indx += 1

    def print_parameters(self, opt_map=None):
        ret = []
        if opt_map is None:
            opt_map = self.id_list
        for indx in range(len(opt_map)):
            ids = opt_map[indx]
            for id in ids:
                gate = id[0]
                par_id = id[1:4]
                chan = par_id[0]
                comp = par_id[1]
                param = par_id[2]
                gate_instr = self.instructions[gate]
                par = gate_instr.comps[chan][comp].params[param]
            nice_id = gate + "-" + "-".join(par_id)
            ret.append(f"{nice_id:28}: {par}\n")
        return "".join(ret)


class Instruction():
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
            channels: list = [],
            t_start: np.float64 = 0.0,
            t_end: np.float64 = 0.0,
            ):
        self.name = name
        self.t_start = t_start
        self.t_end = t_end
        self.comps = {}
        for chan in channels:
            self.comps[chan] = {}
        # TODO remove redundancy of channels in instruction

    def write_config(self):
        cfg = copy.deepcopy(self.__dict__)
        for chan in self.comps:
            for comp in self.comps[chan]:
                cfg['comps'][chan][comp] = 0
        return cfg

    def add_component(self, comp: InstructionComponent, chan: str):
        self.comps[chan][comp.name] = comp


def merge_instructions(gates: list):
    pass


def stack_instructions(gates: list):
    pass