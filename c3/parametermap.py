"""ParameterMap class"""

from typing import List, Dict, Tuple
import hjson
import json
import copy
import numpy as np
import tensorflow as tf
from c3.c3objs import Quantity
from c3.signal.gates import Instruction
from c3.signal.pulse import components as comp_lib
from typing import Union
from tensorflow.errors import InvalidArgumentError


class ParameterMap:
    """
    Collects information about control and model parameters and provides different
    representations depending on use.
    """

    def __init__(
        self, instructions: List[Instruction] = [], generator=None, model=None
    ):
        self.instructions: Dict[str, Instruction] = dict()
        self.opt_map: List[List[Tuple[str]]] = list()
        self.model = model
        self.generator = generator
        for instr in instructions:
            # Is this redundant key necessary?
            self.instructions[instr.name + instr.get_target_str()] = instr

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
        par_ids_model = []
        for comp in self.__components.values():
            for par_name, par_value in comp.params.items():
                par_id = "-".join([comp.name, par_name])
                par_lens[par_id] = par_value.length
                pars[par_id] = par_value
                par_ids_model.append(par_id)

        # Initializing control parameters
        for gate in self.instructions:
            instr = self.instructions[gate]
            for key_elems, par_value in instr.get_optimizable_parameters():
                par_id = "-".join(key_elems)
                par_lens[par_id] = par_value.length
                pars[par_id] = par_value

        self.__par_lens = par_lens
        self.__pars: Dict[str, Quantity] = pars
        self.__par_ids_model = par_ids_model

    def update_parameters(self):
        self.__initialize_parameters()

    def load_values(self, init_point):
        """
        Load a previous parameter point to start the optimization from.

        Parameters
        ----------
        init_point : str
            File location of the initial point

        """
        with open(init_point) as init_file:
            best = json.load(
                init_file
            )  # no hjson: be compatible with optimizer logging

        best_opt_map = best["opt_map"]
        init_p = best["optim_status"]["params"]
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
        self.fromdict(cfg)

    def fromdict(self, cfg: dict) -> None:
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

    def asdict(self, instructions_only=True) -> dict:
        """
        Return a dictionary compatible with config files.
        """
        instructions = {}
        for name, instr in self.instructions.items():
            instructions[name] = instr.asdict()
        if instructions_only:
            return instructions
        else:
            out_dict = dict()
            out_dict["instructions"] = instructions
            out_dict["model"] = self.model.asdict()
            return out_dict

    def __str__(self) -> str:
        return hjson.dumps(self.asdict())

    def get_full_params(self) -> Dict[str, Quantity]:
        """
        Returns the full parameter vector, including model and control parameters.
        """
        return self.__pars

    def get_not_opt_params(self, opt_map=None) -> Dict[str, Quantity]:
        opt_map = self.get_opt_map(opt_map)
        out_dict = copy.copy(self.__pars)
        for equiv_ids in opt_map:
            for key in equiv_ids:
                del out_dict[key]
        return out_dict

    def get_opt_units(self) -> List[str]:
        """
        Returns a list of the units of the optimized quantities.
        """
        units = []
        for equiv_ids in self.get_opt_map():
            key = equiv_ids[0]
            units.append(self.__pars[key].unit)
        return units

    def get_opt_limits(self):
        limits = []
        for equiv_ids in self.get_opt_map():
            key = equiv_ids[0]
            limits.append((self.__pars[key].get_limits()))
        return limits

    def check_limits(self, opt_map):
        for equiv_ids in self.get_opt_map():
            if len(equiv_ids) > 1:
                limit = self.__pars[equiv_ids[0]].get_limits()
                for key in equiv_ids[1:]:
                    assert self.__pars[key].get_limits() == limit

    def get_parameter(self, par_id: Tuple[str, ...]) -> Quantity:
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
        key = "-".join(par_id)
        try:
            value = self.__pars[key]
        except KeyError as ke:
            raise Exception(f"C3:ERROR:Parameter {key} not defined.") from ke
        return value

    def get_parameters(self, opt_map=None) -> List[Quantity]:
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
        opt_map = self.get_opt_map(opt_map)
        for equiv_ids in opt_map:
            key = equiv_ids[0]
            values.append(self.__pars[key])
        return values

    def get_parameter_dict(self, opt_map=None) -> Dict[str, Quantity]:
        """
        Return the current parameters in a dictionary including keys.
        Parameters
        ----------
        opt_map

        Returns
        -------
        Dictionary with Quantities
        """
        value_dict = dict()
        opt_map = self.get_opt_map(opt_map)
        for equiv_ids in opt_map:
            key = equiv_ids[0]
            value_dict[key] = self.__pars[key]
        return value_dict

    def set_parameters(self, values: list, opt_map=None, extend_bounds=False) -> None:
        """Set the values in the original instruction class.

        Parameters
        ----------
        values: list
            List of parameter values. Can be nested, if a parameter is matrix valued.
        opt_map: list
            Corresponding identifiers for the parameter values.
        extend_bounds: bool
            If true bounds of quantity objects will be extended.

        """
        model_updated = False
        val_indx = 0
        opt_map = self.get_opt_map(opt_map)
        assert len(values) == len(
            opt_map
        ), "Different number of elements in values and opt_map"
        for equiv_ids in opt_map:
            for key in equiv_ids:
                # We check if a model parameter has changed
                model_updated = key in self.__par_ids_model or model_updated
                try:
                    par = self.__pars[key]
                except ValueError as ve:
                    raise Exception(f"C3:ERROR:{key} not defined.") from ve
                try:
                    par.set_value(values[val_indx], extend_bounds=extend_bounds)
                except (ValueError, InvalidArgumentError) as ve:
                    try:
                        raise Exception(
                            f"C3:ERROR:Trying to set {key} "
                            f"to value {values[val_indx]} "
                            f"but has to be within {par.offset} .."
                            f" {(par.offset + par.scale)}."
                        ) from ve
                    except TypeError:
                        raise ve
            val_indx += 1

        # TODO: This check is too simple. Not every model parameter requires an update.
        if model_updated and self.model:
            self.model.update_model()

    def get_parameters_scaled(self, opt_map=None) -> np.ndarray:
        """
        Return the current parameters. This fuction should only be called by an
        optimizer. Are you an optimizer?

        Parameters
        ----------
        opt_map: tuple
            Hierarchical identifier for parameters.

        Returns
        -------
        list of Quantity

        """
        values = []
        opt_map = self.get_opt_map(opt_map)
        for equiv_ids in opt_map:
            key = equiv_ids[0]
            par = self.__pars[key]
            values.append(par.get_opt_value())
        # TODO is there a reason to not return a tensorflow array
        return np.concatenate(values, axis=0).flatten()

    def set_parameters_scaled(
        self, values: Union[tf.constant, tf.Variable], opt_map=None
    ) -> None:
        """
        Set the values in the original instruction class. This fuction should only be
        called by an optimizer. Are you an optimizer?

        Parameters
        ----------
        values: list
            List of parameter values. Matrix valued parameters need to be flattened.

        """
        model_updated = False
        val_indx = 0
        opt_map = self.get_opt_map(opt_map)
        for equiv_ids in opt_map:
            key = equiv_ids[0]
            par_len = self.__pars[key].length
            for par_id in equiv_ids:
                key = par_id
                model_updated = True if key in self.__par_ids_model else model_updated
                par = self.__pars[key]
                par.set_opt_value(values[val_indx : val_indx + par_len])
            val_indx += par_len
        if model_updated:
            self.model.update_model()

    def get_key_of_parameters_scaled(self, idx, opt_map=None) -> str:
        opt_map = self.get_opt_map(opt_map)
        curr_indx = 0
        for equiv_ids in opt_map:
            key = equiv_ids[0]
            par_len = self.__pars[key].length
            curr_indx += par_len
            if idx < curr_indx:
                return key
        return None

    def set_opt_map(self, opt_map) -> None:
        """
        Set the opt_map, i.e. which parameters will be optimized.
        """
        opt_map = self.get_opt_map(opt_map)
        for equiv_ids in opt_map:
            for pid in equiv_ids:
                key = pid
                if key not in self.__pars:
                    par_strings = "\n".join(self.__pars.keys())
                    raise Exception(
                        f"C3:ERROR:Parameter {key} not defined in {par_strings}"
                    )
        self.check_limits(opt_map)
        self.opt_map = opt_map

    def get_opt_map(self, opt_map=None) -> List[List[str]]:
        if opt_map is None:
            opt_map = self.opt_map

        for i, equiv_ids in enumerate(opt_map):
            for j, par_id in enumerate(equiv_ids):
                if type(par_id) is str:
                    continue
                key = "-".join(par_id)
                opt_map[i][j] = key

        return opt_map

    def str_parameters(
        self, opt_map: Union[List[List[Tuple[str]]], List[List[str]]] = None
    ) -> str:
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
        opt_map = self.get_opt_map(opt_map)
        ret = []
        for equiv_ids in opt_map:
            par_id = equiv_ids[0]
            key = par_id
            par = self.__pars[key]
            ret.append(f"{key:38}: {par}\n")
            if len(equiv_ids) > 1:
                for eid in equiv_ids[1:]:
                    ret.append(eid)
                    ret.append("\n")
                ret.append("\n")
        return "".join(ret)

    def print_parameters(self, opt_map=None) -> None:
        """
        Print current parameters to stdout.
        """
        opt_map = self.get_opt_map(opt_map)
        print(self.str_parameters(opt_map))
