"""Object that deals with the open loop optimal control."""

import os
import shutil
import time
import tensorflow as tf
from c3.optimizers.optimizer import Optimizer
from c3.utils.utils import log_setup


class C1(Optimizer):
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
    opt_gates : dict
        Dictionary with ideal gate names as values and evaluated gate names as keys.
    """

    def __init__(
        self,
        dir_path,
        fid_func,
        fid_subspace,
        pmap,
        callback_fids=[],
        algorithm=None,
        store_unitaries=False,
        options={},
        run_name=None,
        interactive=True,
        opt_gates=None,
        logger=None,
    ) -> None:
        super().__init__(
            pmap=pmap,
            algorithm=algorithm,
            store_unitaries=store_unitaries,
            logger=logger,
        )
        self.fid_func = fid_func
        self.fid_subspace = fid_subspace
        self.callback_fids = callback_fids
        self.options = options
        self.opt_gates = opt_gates
        self.__dir_path = dir_path
        self.__run_name = run_name
        self.interactive = interactive

    def log_setup(self) -> None:
        """
        Create the folders to store data.
        """
        dir_path = os.path.abspath(self.__dir_path)
        run_name = self.__run_name
        if run_name is None:
            run_name = "c1_" + self.fid_func.__name__ + "_" + self.algorithm.__name__
        self.logdir = log_setup(dir_path, run_name)
        self.logname = "open_loop.log"
        if isinstance(self.exp.created_by, str):
            shutil.copy2(self.exp.created_by, self.logdir)
        if isinstance(self.created_by, str):
            shutil.copy2(self.created_by, self.logdir)

    def load_model_parameters(self, adjust_exp: str) -> None:
        self.pmap.load_values(adjust_exp)
        self.pmap.model.update_model()
        shutil.copy(adjust_exp, os.path.join(self.logdir, "adjust_exp.log"))

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
        if self.opt_gates is not None:
            renamed_propagators = dict()
            for k, v in self.opt_gates.items():
                renamed_propagators[v] = propagators[k]
        else:
            renamed_propagators = propagators
        try:
            goal = self.fid_func(
                U_dict=renamed_propagators,
                index=self.index,
                dims=dims,
                eval=self.evaluation + 1,
            )
        except TypeError as e:
            # If additional information about the experiment is necessary in the fidelity
            # function pass on also the experiment
            try:
                goal = self.fid_func(
                    exp=self.exp,
                    U_dict=renamed_propagators,
                    index=self.index,
                    dims=dims,
                    eval=self.evaluation + 1,
                )
            except TypeError:
                raise e

        with open(self.logdir + self.logname, "a") as logfile:
            logfile.write(f"\nEvaluation {self.evaluation + 1} returned:\n")
            logfile.write("goal: {}: {}\n".format(self.fid_func.__name__, float(goal)))
            for cal in self.callback_fids:
                val = cal(
                    renamed_propagators,
                    self.index,
                    dims,
                    self.logdir,
                    self.evaluation + 1,
                )
                if isinstance(val, tf.Tensor):
                    val = float(val.numpy())
                logfile.write("{}: {}\n".format(cal.__name__, val))
                self.optim_status[cal.__name__] = val
            logfile.flush()

        self.optim_status["params"] = [
            par.numpy().tolist() for par in self.pmap.get_parameters()
        ]
        self.optim_status["goal"] = float(goal)
        self.optim_status["time"] = time.asctime()
        self.evaluation += 1
        return goal
