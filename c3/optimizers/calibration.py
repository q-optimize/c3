"""Object that deals with the closed loop optimal control."""

import os
import time
import hjson
import pickle
import inspect

from c3.c3objs import hjson_decode
from c3.optimizers.optimizer import Optimizer
from c3.libraries.algorithms import algorithms
from c3.utils.utils import log_setup


class Calibration(Optimizer):
    """
    Object that deals with the closed loop optimal control.

    Parameters
    ----------
    dir_path : str
        Filepath to save results
    eval_func : callable
        infidelity function to be minimized
    pmap : ParameterMap
        Identifiers for the parameter vector
    algorithm : callable
        From the algorithm library
    options : dict
        Options to be passed to the algorithm
    run_name : str
        User specified name for the run, will be used as root folder
    """

    def __init__(
        self,
        eval_func,
        pmap,
        algorithm,
        dir_path=None,
        exp_type=None,
        exp_right=None,
        options={},
        run_name=None,
    ):
        if type(algorithm) is str:
            algorithm = algorithms[algorithm]
        super().__init__(pmap=pmap, algorithm=algorithm)
        self.set_eval_func(eval_func, exp_type)
        self.options = options
        self.exp_right = exp_right
        self.__dir_path = dir_path
        self.__run_name = run_name
        self.run = self.optimize_controls  # alias for legacy method

    def set_eval_func(self, eval_func, exp_type):
        """
        Setter for the eval function.

        Parameters
        ----------
        eval_func : callable
            Function to be evaluated

        """
        # TODO: Implement shell for experiment communication
        self.eval_func = eval_func

    def log_setup(self) -> None:
        """
        Create the folders to store data.

        Parameters
        ----------
        dir_path : str
            Filepath
        run_name : str
            User specified name for the run

        """
        run_name = self.__run_name
        if run_name is None:
            run_name = self.eval_func.__name__ + self.algorithm.__name__
        self.logdir = log_setup(self.__dir_path, run_name)
        self.logname = "calibration.log"

        # We create a copy of the source code of the evaluation function in the log
        with open(os.path.join(self.logdir, "eval_func.py"), "w") as eval_source:
            eval_source.write(inspect.getsource(self.eval_func))

    def optimize_controls(self) -> None:
        """
        Apply a search algorithm to your gateset given a fidelity function.
        """
        self.log_setup()
        self.start_log()
        self.picklefilename = self.logdir + "dataset.pickle"
        print(f"C3:STATUS:Saving as: {os.path.abspath(self.logdir + self.logname)}")
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
        with open(os.path.join(self.logdir, "best_point_" + self.logname), "r") as file:
            best_params = hjson.load(file, object_pairs_hook=hjson_decode)[
                "optim_status"
            ]["params"]
        self.pmap.set_parameters(best_params)
        self.end_log()
        measurements = []
        with open(self.picklefilename, "rb") as pickle_file:
            while True:
                try:
                    measurements.append(pickle.load(pickle_file))
                except EOFError:
                    break
        learn_from = {}
        learn_from["seqs_grouped_by_param_set"] = measurements
        learn_from["opt_map"] = self.pmap.opt_map
        with open(self.picklefilename, "wb+") as pickle_file:
            pickle.dump(learn_from, pickle_file)

    def goal_run(self, current_params):
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
        # There could be more processing happening here, i.e. an exp could
        # generate signals for an experiment and send those to eval_func.
        params = self.pmap.get_parameters()

        goal, results, results_std, seqs, shots = self.eval_func(params)
        self.optim_status["params"] = [
            par.numpy().tolist() for par in self.pmap.get_parameters()
        ]
        self.optim_status["goal"] = float(goal)
        self.optim_status["time"] = time.asctime()
        self.evaluation += 1
        self.log_pickle(params, seqs, results, results_std, shots)
        return goal

    def log_pickle(self, params, seqs, results, results_std, shots):
        """
        Save a pickled version of the performed experiment, suitable for model learning.

        Parameters
        ----------
        params : tf.Tensor
            Vector of parameter values
        seqs : list
            Strings identifying the performed instructions
        results : list
            Values of the goal function
        results_std : list
            Standard deviation of the results, in the case of noisy data
        shots : list
            Number of repetitions used in averaging noisy data

        """
        data_entry = {
            "params": params,
            "seqs": seqs,
            "results": results,
            "results_std": results_std,
            "shots": shots,
        }
        with open(self.picklefilename, "ab") as file:
            pickle.dump(data_entry, file)
