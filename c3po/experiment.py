"""Experiment class that models the whole experiment."""

import types
import numpy as np
import tensorflow as tf
from c3po.component import C3obj


class Experiment:
    """
    It models all of the behaviour of the physical experiment.

    It contains boxes that perform a part of the experiment routine.

    Parameters
    ----------
    model: Model
    generator: Generator

    """

    def __init__(self, model, generator):
        self.model = model
        self.generator = generator

    def list_parameters(self):
        par_list = []
        par_list.extend(self.model.list_parameters())
        devices = self.generator.devices
        for key in devices:
            par_list.extend(devices[key].list_parameters())
        return par_list

    def parameter_indeces(self, opt_map: list):
        par_list = self.list_parameters()
        par_indx = []
        for par_id in opt_map:
            par_indx.append(par_list.index(par_id))
        return par_indx

    def get_parameters(self, opt_map=None):
        """Return list of values and bounds of parameters in opt_map."""
        if opt_map is None:
            opt_map = self.list_parameters()
        values = []
        values.append(self.model.get_parameters())
        devices = self.generator.devices
        for key in devices:
            pars = devices[key].get_parameters()
            if not (pars == []):
                values.append(pars)
        # TODO Deal with bounds correctly
        self.par_lens = [len(list) for list in values]
        par_indx = self.parameter_indeces(opt_map)
        values_flat = []
        for list in values:
            values_flat.extend(list)
        values_new = [values_flat[indx] for indx in par_indx]
        bounds = np.kron(np.array([[0.9], [1.1]]), np.array(values_new)).T
        return values_new, bounds

    def set_parameters(self, values: list, opt_map: list):
        """Set the values in the original instruction class."""
        pars, _ = self.get_parameters(self.list_parameters())
        par_indx = self.parameter_indeces(opt_map)
        indx = 0
        for par_ii in par_indx:
            pars[par_ii] = values[indx]
            indx = indx + 1
        first_ind = 0
        last_ind = first_ind + self.par_lens[0]
        params = pars[first_ind:last_ind]
        self.model.set_parameters(params)
        devs = self.generator.devices
        indx = 0
        for par in opt_map:
            if par[0] in devs.keys():
                devs[par[0]].params[par[1]] = values[indx]
            indx += 1


class Measurement:
    """
    It models all of the behaviour of the measurement process.

    It includes initialization and readout errors.

    Parameters
    ----------
    Tasks: list

    """

    def __init__(self, tasks):
        self.tasks = {}
        for task in tasks:
            self.tasks[task.name] = task

    def measure(self, U_dict):
        with tf.name_scope('Measurement Tasks'):
            init_ground = self.devices["init_ground"]
            fidelity = self.devices["fidelity"]
            meas_err = self.devices["meas_err"]
            evolution = self.devices["evolution"]
            psi_init = init_ground.initialise()
            psi_final = evolution.evolve(U_dict, psi_init)
            measured = meas_err.measure(psi_final)
            fid = fidelity.process(measured)
        return fid


class Task(C3obj):
    """Task that is part of the measurement setup."""

    def __init__(
            self,
            name: str = " ",
            desc: str = " ",
            comment: str = " ",
    ):
        super().__init__(
            name=name,
            desc=desc,
            comment=comment
        )
        self.params = {}

    def get_parameters(self):
        params = []
        for key in sorted(self.params.keys()):
            params.append(self.params[key])
        return params

    def set_parameters(self, values):
        idx = 0
        for key in sorted(self.params.keys()):
            self.params[key] = values[idx]
            idx += 1

    def list_parameters(self):
        par_list = []
        for par_key in sorted(self.params.keys()):
            par_id = (self.name, par_key)
            par_list.append(par_id)
        return par_list


class InitialiseGround(Task):
    """Initialise the ground state with a given thermal distribution."""

    def __init__(
            self,
            name: str = "",
            desc: str = " ",
            comment: str = " ",
            temp: np.float64 = 0.0
    ):
        super().__init__(
            name=name,
            desc=desc,
            comment=comment
        )
        self.params['temp'] = temp

    def initialise(self):
        # init_state = thermal population with self.temp
        # return init_state
        pass


class MeasureExpectationZ(Task):
    """Initialise the ground state with a given thermal distribution."""

    def __init__(
            self,
            name: str = "",
            desc: str = " ",
            comment: str = " ",
            meas_error_matrix: np.array = np.array([[1.0, 0.0], [0.0, 1.0]])
    ):
        super().__init__(
            name=name,
            desc=desc,
            comment=comment
        )
        self.params['meas_error'] = np.flat(meas_error_matrix)

    def measure(self):
        # init_state = thermal population with self.temp
        # return init_state
        pass


class Fidelity(Task):
    """Perform the fidelity measurement."""

    def __init__(
            self,
            name: str = "",
            desc: str = " ",
            comment: str = " ",
            fidelity_fct: types.FunctionType = None
    ):
        super().__init__(
            name=name,
            desc=desc,
            comment=comment
        )
        self.fidelity_fct = fidelity_fct

    def process(self, measured):
        # return self.fidelity_fct(measured)
        pass


class Evolution(Task):
    """Evolve initial state using U_dict."""

    def __init__(
            self,
            name: str = "",
            desc: str = " ",
            comment: str = " ",
            evolution_fct: types.FunctionType = None
    ):
        super().__init__(
            name=name,
            desc=desc,
            comment=comment
        )
        self.evolution_fct = evolution_fct

    def evolve(self, U_dict, psi_init):
        # return self.evolution_fct(U_dict, psi_init)
        pass
