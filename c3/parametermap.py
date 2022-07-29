"""ParameterMap class"""

from typing import List, Dict, Tuple
import hjson
import copy
import numpy as np
import tensorflow as tf
from c3.c3objs import Quantity, hjson_decode, hjson_encode
from c3.signal.gates import Instruction
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
            self.instructions[instr.get_key()] = instr

        # Collecting model components
        components = {}
        if model:
            components.update(model.couplings)
            components.update(model.subsystems)
            components.update(model.tasks)
        if generator:
            components.update(generator.devices)
        self.__components = components
        self.update_model = False
        self.set_parameters_scaled = self._set_parameters_scaled_ctrls
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

        self._par_lens = par_lens
        self._pars: Dict[str, Quantity] = pars
        self._par_ids_model = par_ids_model

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
            best = hjson.load(init_file, object_pairs_hook=hjson_decode)

        best_opt_map = best["opt_map"]
        init_p = best["optim_status"]["params"]
        self.set_parameters(init_p, best_opt_map, extend_bounds=True)

    def store_values(self, path: str, optim_status=None) -> None:
        """
        Write current parameter values to file. Stores the numeric values, as well as the names
        in form of the opt_map and physical units. If an optim_status is given that will be
        used.

        Parameters
        ----------
        path : str
            Location of the resulting logfile.
        optim_status: dict
            Dictionary containing current parameters and goal function value.
        """
        if optim_status is None:
            optim_status = {
                "params": [par.numpy().tolist() for par in self.get_parameters()]
            }
        with open(path, "w") as value_file:
            val_dict = {
                "opt_map": self.get_opt_map(),
                "units": self.get_opt_units(),
                "optim_status": optim_status,
            }
            value_file.write(hjson.dumps(val_dict, default=hjson_encode))
            value_file.write("\n")

    def read_config(self, filepath: str) -> None:
        """
        Load a file and parse it to create a ParameterMap object.

        Parameters
        ----------
        filepath : str
            Location of the configuration file

        """
        with open(filepath, "r") as cfg_file:
            cfg = hjson.loads(cfg_file.read(), object_pairs_hook=hjson_decode)
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
                # TODO: initialize directly by using the constructor.
                instr = Instruction()
                instr.from_dict(gate, name=key)
            self.instructions[key] = instr
            self.__initialize_parameters()

    def write_config(self, filepath: str) -> None:
        """
        Write dictionary to a HJSON file.
        """
        with open(filepath, "w") as cfg_file:
            hjson.dump(self.asdict(), cfg_file, default=hjson_encode)

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
        return hjson.dumps(self.asdict(), default=hjson_encode)

    def get_full_params(self) -> Dict[str, Quantity]:
        """
        Returns the full parameter vector, including model and control parameters.
        """
        return self._pars

    def get_not_opt_params(self, opt_map=None) -> Dict[str, Quantity]:
        opt_map = self.get_opt_map(opt_map)
        out_dict = copy.copy(self._pars)
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
            units.append(self._pars[key].unit)
        return units

    def get_opt_limits(self):
        limits = []
        for equiv_ids in self.get_opt_map():
            key = equiv_ids[0]
            limits.append((self._pars[key].get_limits()))
        return limits

    def check_limits(self, opt_map):
        """
        Check if all elements of equal ids have the same limits. This has to be checked against if setting values optimizer friendly.

        Parameters
        ----------
        opt_map

        Returns
        -------

        """
        for equiv_ids in self.get_opt_map():
            if len(equiv_ids) > 1:
                limit = self._pars[equiv_ids[0]].get_limits()
                for key in equiv_ids[1:]:
                    if not self._pars[key].get_limits() == limit:
                        raise Exception(
                            "C3:Error:Limits for {key} are not equivalent to {equiv_ids}."
                        )

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
            value = self._pars[key]
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
            values.append(self._pars[key])
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
            value_dict[key] = self._pars[key]
        return value_dict

    def set_parameters(
        self, values: Union[List, np.ndarray], opt_map=None, extend_bounds=False
    ) -> None:
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
        if not len(values) == len(opt_map):
            raise Exception(
                f"C3:Error: Different number of elements in values and opt_map. {len(values)} vs {len(opt_map)}"
            )
        for equiv_ids in opt_map:
            for key in equiv_ids:
                # We check if a model parameter has changed
                model_updated = key in self._par_ids_model or model_updated
                try:
                    par = self._pars[key]
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

    def get_parameters_scaled(self, opt_map=None) -> tf.Tensor:
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
            par = self._pars[key]
            values.append(par.get_opt_value())
        return tf.concat(values, axis=0)

    def _set_parameters_scaled_ctrls(
        self, values: Union[tf.constant, tf.Variable], opt_map=None
    ) -> None:
        """
        Set the values in the original instruction class. This fuction should only be
        called by an optimizer. Are you an optimizer? This method only sets control
        parameters and does not trigger a model update.

        Parameters
        ----------
        values: list
            List of parameter values. Matrix valued parameters need to be flattened.

        """
        val_indx = 0
        opt_map = self.get_opt_map(opt_map)
        for equiv_ids in opt_map:
            key = equiv_ids[0]
            par_len = self._pars[key].length
            for par_id in equiv_ids:
                key = par_id
                par = self._pars[key]
                par.set_opt_value(values[val_indx : val_indx + par_len])
            val_indx += par_len

    def _set_parameters_scaled_model(
        self, values: Union[tf.constant, tf.Variable], opt_map=None
    ) -> None:
        """
        Set the values in the original instruction class. This fuction should only be
        called by an optimizer. Are you an optimizer? Also update the model.

        Parameters
        ----------
        values: list
            List of parameter values. Matrix valued parameters need to be flattened.

        """
        self._set_parameters_scaled_ctrls(values)
        self.model.update_model()

    def get_key_from_scaled_index(self, idx, opt_map=None) -> str:
        """
        Get the key of the value at position `ìdx` of the scaled_parameters output
        Parameters
        ----------
        idx
        opt_map

        Returns
        -------

        """
        opt_map = self.get_opt_map(opt_map)
        curr_indx = 0
        for equiv_ids in opt_map:
            key = equiv_ids[0]
            par_len = self._pars[key].length
            curr_indx += par_len
            if idx < curr_indx:
                return key
        return ""

    def set_opt_map(self, opt_map) -> None:
        """
        Set the opt_map, i.e. which parameters will be optimized.
        """
        opt_map = self.get_opt_map(opt_map)
        update_model = False
        for equiv_ids in opt_map:
            for pid in equiv_ids:
                key = pid
                if key not in self._pars:
                    par_strings = "\n".join(self._pars.keys())
                    raise Exception(
                        f"C3:ERROR:Parameter {key} not defined in {par_strings}"
                    )
                update_model = key in self._par_ids_model or update_model
        self.set_update_model(update_model)
        self.check_limits(opt_map)
        self.opt_map = opt_map

    def set_update_model(self, update: bool) -> None:
        self.update_model = update
        if update:
            self.set_parameters_scaled = self._set_parameters_scaled_model
        else:
            self.set_parameters_scaled = self._set_parameters_scaled_ctrls

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
        self,
        opt_map: Union[List[List[Tuple[str]]], List[List[str]]] = None,
        human=False,
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
            par = self._pars[key]
            if human and par.length > 4:  # Don't print large num of values
                ret.append(f"{key:38}: <{par.length} values>\n")
            else:
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
        print(self.str_parameters(opt_map, human=True))
