"""Object that deals with the open loop optimal control."""

import os
import shutil
import tensorflow as tf
from typing import Callable, List

from c3.optimizers.optimizer import Optimizer
from c3.utils.utils import log_setup

from c3.libraries.algorithms import algorithms
from c3.libraries.fidelities import fidelities


class OptimalControl(Optimizer):
    """
    Object that deals with the open loop optimal control.

    Parameters
    ----------
    dir_path : str
        Filepath to save results
    fid_func : callable
        infidelity function to be minimized
    fid_subspace : list
        Indeces identifying the subspace to be compared
    pmap : ParameterMap
        Identifiers for the parameter vector
    callback_fids : list of callable
        Additional fidelity function to be evaluated and stored for reference
    algorithm : callable
        From the algorithm library
        Save plots of control signals
    store_unitaries : boolean
        Store propagators as text and pickle
    options : dict
        Options to be passed to the algorithm
    run_name : str
        User specified name for the run, will be used as root folder
    fid_func_kwargs: dict
        Additional kwargs to be passed to the main fidelity function.
    """

    def __init__(
        self,
        fid_func,
        fid_subspace,
        pmap,
        dir_path=None,
        callback_fids=None,
        algorithm=None,
        initial_point: str = "",
        store_unitaries=False,
        options={},
        run_name=None,
        interactive=True,
        include_model=False,
        logger=None,
        fid_func_kwargs={},
    ) -> None:
        if type(algorithm) is str:
            algorithm = algorithms[algorithm]
        super().__init__(
            pmap=pmap,
            algorithm=algorithm,
            initial_point=initial_point,
            store_unitaries=store_unitaries,
            logger=logger,
        )
        self.set_fid_func(fid_func)
        self.callback_fids: List[Callable] = []
        if callback_fids:
            self.set_callback_fids(callback_fids)
        self.fid_subspace = fid_subspace
        self.options = options
        self.__dir_path = dir_path
        self.__run_name = run_name
        self.interactive = interactive
        self.update_model = include_model
        self.fid_func_kwargs = fid_func_kwargs
        self.run = (
            self.optimize_controls
        )  # Alias the legacy name for the method running the
        # optimization

    def set_fid_func(self, fid_func) -> None:
        if type(fid_func) is str:
            if "lindbladian" in self.pmap.model.frame:
                fid = "lindbladian_" + fid_func
            else:
                fid = fid_func
            try:
                self.fid_func = fidelities[fid]
            except KeyError:
                raise Exception(f"C3:ERROR:Unkown goal function: {fid} ")
            print(f"C3:STATUS:Found {fid} in libraries.")
        else:
            self.fid_func = fid_func

    def set_callback_fids(self, callback_fids) -> None:
        if "lindbladian" in self.pmap.model.frame:
            cb_fids = ["lindbladian_" + f for f in callback_fids]
        else:
            cb_fids = callback_fids
        for cb_fid in cb_fids:
            try:
                cb_fid_func = fidelities[cb_fid]
            except KeyError:
                raise Exception(f"C3:ERROR:Unkown goal function: {cb_fid}")
            print(f"C3:STATUS:Found {cb_fid} in libraries.")
            self.callback_fids.append(cb_fid_func)

    def log_setup(self) -> None:
        """
        Create the folders to store data.
        """
        run_name = self.__run_name
        if run_name is None:
            run_name = "c1_" + self.fid_func.__name__ + "_" + self.algorithm.__name__
        self.logdir = log_setup(self.__dir_path, run_name)
        self.logname = "open_loop.c3log"
        if isinstance(self.exp.created_by, str):
            shutil.copy2(self.exp.created_by, self.logdir)
        if isinstance(self.created_by, str):
            shutil.copy2(self.created_by, self.logdir)

    def load_model_parameters(self, adjust_exp: str) -> None:
        self.pmap.load_values(adjust_exp)
        self.pmap.model.update_model()
        shutil.copy(adjust_exp, os.path.join(self.logdir, "adjust_exp.c3log"))

    def optimize_controls(self, setup_log: bool = True) -> None:
        """
        Apply a search algorithm to your gateset given a fidelity function.
        """
        if setup_log:
            self.log_setup()
        self.start_log()
        self.exp.set_enable_store_unitaries(self.store_unitaries, self.logdir)
        print(f"C3:STATUS:Saving as: {os.path.abspath(self.logdir + self.logname)}")
        index = []
        for name in self.fid_subspace:
            index.append(self.pmap.model.names.index(name))
        self.index = index
        x_init = self.pmap.get_parameters_scaled()
        try:
            self.algorithm(
                x_init,
                fun=self.fct_to_min,
                fun_grad=self.fct_to_min_autograd,
                grad_lookup=self.lookup_gradient,
                options=self.options,
            )
        except KeyboardInterrupt:
            pass
        self.load_best(self.logdir + "best_point_" + self.logname)
        self.end_log()

    @tf.function
    def goal_run(self, current_params: tf.Tensor) -> tf.float64:
        """
        Evaluate the goal function for current parameters.

        Parameters
        ----------
        current_params : tf.Tensor
            Vector representing the current parameter values.

        Returns
        -------
        tf.float64
            Value of the goal function
        """
        self.pmap.set_parameters_scaled(current_params)
        dims = self.pmap.model.dims
        propagators = self.exp.compute_propagators()

        goal = self.fid_func(
            propagators=propagators,
            instructions=self.pmap.instructions,
            index=self.index,
            dims=dims,
            n_eval=self.evaluation + 1,
            **self.fid_func_kwargs,
        )
        self.evaluation += 1
        return goal
