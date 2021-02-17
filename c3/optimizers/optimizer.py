"""Optimizer object, where the optimal control is done."""

import os
import time
from typing import Callable, Union

import numpy as np
import tensorflow as tf
import json
import c3.libraries.algorithms as algorithms


class Optimizer:
    """
    General optimizer class from which specific classes are inherited.

    Parameters
    ----------
    algorithm : callable
        From the algorithm library
    plot_dynamics : boolean
        Save plots of time-resolved dynamics in dir_path
    plot_pulses : boolean
        Save plots of control signals
    store_unitaries : boolean
        Store propagators as text and pickle
    """

    def __init__(
        self,
        pmap,
        algorithm=None,
        store_unitaries=False,
    ):
        self.pmap = pmap
        self.optim_status = {}
        self.gradients = {}
        self.current_best_goal = 9876543210.123456789
        self.current_best_params = None
        self.evaluation = 0
        self.store_unitaries = store_unitaries
        self.created_by = None
        self.logname = None
        self.options = None
        self.dir_path = None
        self.logdir = None
        self.set_algorithm(algorithm)

    def set_algorithm(self, algorithm: Callable) -> None:
        if algorithm:
            self.algorithm = algorithm
        else:
            print("C3:WARNING:No algorithm passed. Using default LBFGS")
            self.algorithm = algorithms.lbfgs

    def replace_logdir(self, new_logdir):
        """
        Specify a new filepath to store the log.

        Parameters
        ----------
        new_logdir

        """
        old_logdir = self.logdir
        self.logdir = new_logdir
        try:
            os.remove(os.path.join(self.dir_path, "recent"))
        except FileNotFoundError:
            pass
        # os.remove(self.dir_path + self.string)
        try:
            os.rmdir(old_logdir)
        except OSError:
            pass

    def set_exp(self, exp) -> None:
        self.exp = exp

    def set_created_by(self, config) -> None:
        """
        Store the config file location used to created this optimizer.
        """
        self.created_by = config

    def load_best(self, init_point) -> None:
        """
        Load a previous parameter point to start the optimization from. Legacy wrapper.
        Method moved to Parametermap.

        Parameters
        ----------
        init_point : str
            File location of the initial point

        """
        self.pmap.load_values(init_point)

    def start_log(self) -> None:
        """
        Initialize the log with current time.

        """
        self.start_time = time.time()
        start_time_str = str(f"{time.asctime(time.localtime())}\n\n")
        with open(self.logdir + self.logname, "a") as logfile:
            logfile.write("Starting optimization at ")
            logfile.write(start_time_str)
            logfile.write("Optimization parameters:\n")
            logfile.write(json.dumps(self.pmap.opt_map))
            logfile.write("\n")
            logfile.write("Units:\n")
            logfile.write(json.dumps(self.pmap.get_opt_units()))
            logfile.write("\n")
            logfile.write("Algorithm options:\n")
            logfile.write(json.dumps(self.options))
            logfile.write("\n")
            logfile.flush()

    def end_log(self) -> None:
        """
        Finish the log by recording current time and total runtime.

        """
        self.end_time = time.time()
        with open(self.logdir + self.logname, "a") as logfile:
            logfile.write(f"Finished at {time.asctime(time.localtime())}\n")
            logfile.write(f"Total runtime: {self.end_time-self.start_time}\n\n")
            logfile.flush()

    def log_best_unitary(self) -> None:
        """
        Save the best unitary in the log.
        """
        with open(self.logdir + "best_point_" + self.logname, "w") as best_point:
            U_dict = self.exp.propagators
            for gate, U in U_dict.items():
                best_point.write("\n")
                best_point.write(f"Re {gate}: \n")
                best_point.write(f"{np.round(np.real(U), 3)}\n")
                best_point.write("\n")
                best_point.write(f"Im {gate}: \n")
                best_point.write(f"{np.round(np.imag(U), 3)}\n")

    def log_parameters(self) -> None:
        """
        Log the current status. Write parameters to log. Update the current best
        parameters. Call plotting functions as set up.

        """
        if self.optim_status["goal"] < self.current_best_goal:
            self.current_best_goal = self.optim_status["goal"]
            self.current_best_params = self.optim_status["params"]
            with open(self.logdir + "best_point_" + self.logname, "w") as best_point:
                best_dict = {
                    "opt_map": self.pmap.opt_map,
                    "units": self.pmap.get_opt_units(),
                    "optim_status": self.optim_status,
                }
                best_point.write(json.dumps(best_dict))
                best_point.write("\n")
        if self.store_unitaries:
            self.exp.store_Udict(self.optim_status["goal"])
            self.exp.store_unitaries_counter += 1
        with open(self.logdir + self.logname, "a") as logfile:
            logfile.write(
                f"\nFinished evaluation {self.evaluation} at {time.asctime()}\n"
            )
            # logfile.write(json.dumps(self.optim_status, indent=2))
            logfile.write(json.dumps(self.optim_status))
            logfile.write("\n")
            logfile.flush()

    def goal_run(
        self, current_params: Union[np.ndarray, tf.constant]
    ) -> Union[np.ndarray, tf.constant]:
        """
        Placeholder for the goal function. To be implemented by inherited classes.
        """
        return 0

    def goal_run_with_grad(self, current_params):
        with tf.GradientTape() as t:
            t.watch(current_params)
            goal = self.goal_run(current_params)
        grad = t.gradient(goal, current_params)
        return goal, grad

    def lookup_gradient(self, x):
        """
        Return the stored gradient for a given parameter set.

        Parameters
        ----------
        x : np.array
            Parameter set.

        Returns
        -------
        np.array
            Value of the gradient.
        """
        key = str(x)
        return self.gradients.pop(key)

    def fct_to_min(
        self, input_parameters: Union[np.ndarray, tf.constant]
    ) -> Union[np.ndarray, tf.constant]:
        """
        Wrapper for the goal function.

        Parameters
        ----------
        x : [np.array, tf.constant]
            Vector of parameters in the optimizer friendly way.

        Returns
        -------
        [float, tf.constant]
            Value of the goal function. Float if input is np.array else tf.constant
        """

        if isinstance(input_parameters, np.ndarray):
            current_params = tf.constant(input_parameters)
            goal = self.goal_run(current_params)
            self.log_parameters()
            goal = float(goal)
            return goal
        else:
            current_params = input_parameters
            goal = self.goal_run(current_params)
            self.log_parameters()
            return goal

    def fct_to_min_autograd(self, x):
        """
         Wrapper for the goal function, including evaluation and storage of the
         gradient.

        Parameters
         ----------
         x : np.array
             Vector of parameters in the optimizer friendly way.

         Returns
         -------
         float
             Value of the goal function.
        """
        current_params = tf.constant(x)
        goal, grad = self.goal_run_with_grad(current_params)
        if isinstance(grad, tf.Tensor):
            grad = grad.numpy()
        gradients = grad.flatten()
        self.gradients[str(current_params.numpy())] = gradients
        self.optim_status["gradient"] = gradients.tolist()
        self.log_parameters()
        if isinstance(goal, tf.Tensor):
            goal = float(goal)
        return goal
