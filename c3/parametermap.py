"""ParameterMap class"""

from typing import List, Dict, Tuple, Any
import hjson
import copy
import numpy as np
from c3.c3objs import Quantity
from c3.signal.gates import Instruction
from c3.signal.pulse import components as comp_lib


class ParameterMap:
    """
    Collects information about control and model parameters and provides different representations
    depending on use.
    """

    def __init__(self, instructions: list = [], generator=None, model=None):
        self.instructions = {}
        self.opt_map: List = []
        self.model = model
        self.generator = generator
        for instr in instructions:
            self.instructions[instr.name] = instr

        # Collecting model components
        components = {}
        if model:
            components.update(model.couplings)
            components.update(model.subsystems)
            components.update(model.tasks)
        if generator:
            components.update(generator.devices)
        self.__components = components
        self.__initialize_parameters()

    def __initialize_parameters(self) -> None:
        par_lens = {}
        pars = {}
        for comp in self.__components.values():
            for par_name, par_value in comp.params.items():
                par_id = (comp.name, par_name)
                par_lens[par_id] = par_value.length
                pars[par_id] = par_value

        # Initializing control parameters
        for gate in self.instructions:
            instr = self.instructions[gate]
            for chan in instr.comps.keys():
                for comp in instr.comps[chan]:
                    for par_name, par_value in instr.comps[chan][comp].params.items():
                        par_id = (gate, chan, comp, par_name)
                        par_lens[par_id] = par_value.length
                        pars[par_id] = par_value

        self.__par_lens = par_lens
        self.__pars = pars
            
    def load_values(self, init_point):
        """
        Load a previous parameter point to start the optimization from.

        Parameters
        ----------
        init_point : str
            File location of the initial point

        """
        with open(init_point) as init_file:
            best = hjson.load(init_file)

        best_opt_map = [
            [tuple(par) for par in pset] for pset in best["opt_map"]
        ]
        init_p = best['optim_status']['params']
        self.set_parameters(init_p, best_opt_map)

    def read_config(self, filepath: str) -> None:
        """
        Load a file and parse it to create a ParameterMap object.

        Parameters
        ----------
        filepath : str
            Location of the configuration file

        """
        with open(filepath, "r") as cfg_file:
            cfg = hjson.loads(cfg_file.read())
        for key, gate in cfg.items():
            if "mapto" in gate.keys():
                instr = copy.deepcopy(self.instructions[gate["mapto"]])
                instr.name = key
                for drive_chan, comps in gate["drive_channels"].items():
                    for comp, props in comps.items():
                        for par, val in props["params"].items():
                            instr.comps[drive_chan][comp].params[par].set_value(val)
            else:
                instr = Instruction(
                    name=key,
                    t_start=0.0,
                    t_end=gate["gate_length"],
                    channels=list(gate["drive_channels"].keys()),
                )
                for drive_chan, comps in gate["drive_channels"].items():
                    for comp, props in comps.items():
                        ctype = props.pop("c3type")
                        instr.add_component(
                            comp_lib[ctype](name=comp, **props), chan=drive_chan
                        )
            self.instructions[key] = instr
            self.__initialize_parameters()

    def write_config(self, filepath: str) -> None:
        """
        Write dictionary to a HJSON file.
        """
        with open(filepath, "w") as cfg_file:
            hjson.dump(self.asdict(), cfg_file)

    def asdict(self) -> dict:
        """
        Return a dictionary compatible with config files.
        """
        instructions = {}
        for name, instr in self.instructions.items():
            instructions[name] = instr.asdict()
        return instructions

    def __str__(self) -> str:
        return hjson.dumps(self.asdict())

    def get_full_params(self) -> Dict[str, Quantity]:
        """
        Returns the full parameter vector, including model and control parameters.
        """
        return self.__pars

    def get_opt_units(self) -> List[str]:
        """
        Returns a list of the units of the optimized quantities.
        """
        units = []
        for equiv_ids in self.opt_map:
            units.append(self.__pars[equiv_ids[0]].unit)
        return units

    def get_parameter(self, par_id: Tuple[str]) -> Quantity:
        """
        Return one the current parameters.

        Parameters
        ----------
        par_id: tuple
            Hierarchical identifier for parameter.

        Returns
        -------
        Quantity

        """
        try:
            value = self.__pars[par_id]
        except KeyError as ke:
            for id in self.__pars:
                if id[0] == par_id[0]:
                    print(f"Found {id[0]}.")
            raise Exception(f"C3:ERROR:Parameter {par_id} not defined.") from ke
        return value

    def get_parameters(self) -> List[Quantity]:
        """
        Return the current parameters.

        Parameters
        ----------
        opt_map: list
            Hierarchical identifier for parameters.

        Returns
        -------
        list of Quantity

        """
        values = []
        for equiv_ids in self.opt_map:
            try:
                values.append(self.__pars[equiv_ids[0]])
            except KeyError as ke:
                for par_id in self.__pars:
                    if par_id[0] == equiv_ids[0][0]:
                        print(f"Found {par_id[0]}.")
                raise Exception(
                    f"C3:ERROR:Parameter {equiv_ids[0]} not defined."
                ) from ke
        return values

    def set_parameters(self, values: list, opt_map=None) -> None:
        """Set the values in the original instruction class.

        Parameters
        ----------
        values: list
            List of parameter values. Can be nested, if a parameter is matrix valued.
        opt_map: list
            Corresponding identifiers for the parameter values.

        """
        val_indx = 0
        if opt_map is None:
            opt_map = self.opt_map
        for equiv_ids in opt_map:
            for id in equiv_ids:
                try:
                    par = self.__pars[id]
                except ValueError as ve:
                    raise Exception(f"C3:ERROR:{id} not defined.") from ve
                try:
                    par.set_value(values[val_indx])
                except ValueError as ve:
                    raise Exception(
                        f"C3:ERROR:Trying to set {'-'.join(id)} to value {values[val_indx]} "
                        f"but has to be within {par.offset:.3} .. {(par.offset + par.scale):.3}."
                    ) from ve
            val_indx += 1

    def get_parameters_scaled(self) -> np.ndarray:
        """
        Return the current parameters. This fuction should only be called by an optimizer. Are you
        an optimizer?

        Parameters
        ----------
        opt_map: tuple
            Hierarchical identifier for parameters.

        Returns
        -------
        list of Quantity

        """
        values = []
        for equiv_ids in self.opt_map:
            par = self.__pars[equiv_ids[0]]
            values.append(par.get_opt_value())
        return np.array(values).flatten()

    def set_parameters_scaled(self, values: list) -> None:
        """
        Set the values in the original instruction class. This fuction should only be called by
        an optimizer. Are you an optimizer?

        Parameters
        ----------
        values: list
            List of parameter values. Matrix valued parameters need to be flattened.
        opt_map: list
            Corresponding identifiers for the parameter values.

        """
        val_indx = 0
        for equiv_ids in self.opt_map:
            par_len = self.__pars[equiv_ids[0]].length
            for id in equiv_ids:
                par = self.__pars[id]
                par.set_opt_value(values[val_indx : val_indx + par_len])
            val_indx += par_len

    def set_opt_map(self, opt_map) -> None:
        """
        Set the opt_map, i.e. which parameters will be optimized.
        """
        for equiv_ids in opt_map:
            for pid in equiv_ids:
                if not pid in self.__pars:
                    raise Exception(f"C3:ERROR:Parameter {pid} not defined.")
        self.opt_map = opt_map

    # TODO: F811 redefinition of unused '__str__' from line 111
    def __str__(self) -> str:  # type: ignore
        """
        Return a multi-line human-readable string of all defined parameter names and
        current values.

        Returns
        -------
        str
            Parameters and their values
        """
        ret = []

        for par_id, par in self.__pars.items():
            nice_id = "-".join(par_id)
            ret.append(f"{nice_id:38}: {par}\n")

        return "".join(ret)

    def str_parameters(self, opt_map) -> str:
        """
        Return a multi-line human-readable string of the optmization parameter names and
        current values.

        Parameters
        ----------
        opt_map: list
            Optionally use only the specified parameters.

        Returns
        -------
        str
            Parameters and their values
        """
        ret = []
        for equiv_ids in opt_map:
            par_id = equiv_ids[0]
            par = self.__pars[equiv_ids[0]]
            nice_id = "-".join(par_id)
            ret.append(f"{nice_id:38}: {par}\n")
            if len(equiv_ids) > 1:
                for eid in equiv_ids[1:]:
                    ret.append("-".join(eid))
                    ret.append("\n")
                ret.append("\n")
        return "".join(ret)

    def print_parameters(self):
        """
        Print current parameters to stdout.
        """
        print(self.str_parameters(self.opt_map))
