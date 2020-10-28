"""Optimizer object, where the optimal control is done."""

import os
import time
import json
import tensorflow as tf
import numpy as np
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
        algorithm=None,
        plot_dynamics=False,
        plot_pulses=False,
        store_unitaries=False,
    ):

        self.optim_status = {}
        self.gradients = {}
        self.current_best_goal = 9876543210.123456789
        self.current_best_params = None
        self.evaluation = 0
        self.plot_dynamics = plot_dynamics
        self.plot_pulses = plot_pulses
        self.store_unitaries = store_unitaries
        if algorithm is not None:
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
        os.remove(self.dir_path + "/recent")
        # os.remove(self.dir_path + self.string)
        os.rmdir(old_logdir)

    def set_exp(self, exp):
        self.exp = exp

    def start_log(self):
        """
        Initialize the log with current time.

        """
        self.start_time = time.time()
        start_time_str = str(f"{time.asctime(time.localtime())}\n\n")
        with open(self.logdir + self.logname, "a") as logfile:
            logfile.write("Starting optimization at ")
            logfile.write(start_time_str)
            logfile.write("Optimization parameters:\n")
            logfile.write(json.dumps(self.opt_map))
            logfile.write("\n\n")
            logfile.write("Algorithm options:\n")
            logfile.write(json.dumps(self.options))
            logfile.write("\n")
            logfile.flush()

    def end_log(self):
        """
        Finish the log by recording current time and total runtime.

        """
        self.end_time = time.time()
        with open(self.logdir + self.logname, "a") as logfile:
            logfile.write(f"Finished at {time.asctime(time.localtime())}\n")
            logfile.write(f"Total runtime: {self.end_time-self.start_time}\n\n")
            logfile.flush()

    def log_best_unitary(self):
        """
        Save the best unitary in the log.
        """
        with open(self.logdir + "best_point_" + self.logname, "w") as best_point:
            U_dict = self.exp.unitaries
            for gate, U in U_dict.items():
                best_point.write("\n")
                best_point.write(f"Re {gate}: \n")
                best_point.write(f"{np.round(np.real(U), 3)}\n")
                best_point.write("\n")
                best_point.write(f"Im {gate}: \n")
                best_point.write(f"{np.round(np.imag(U), 3)}\n")

    def log_parameters(self):
        """
        Log the current status. Write parameters to log. Update the current best parameters. Call plotting functions as
        set up.

        """
        if self.optim_status["goal"] < self.current_best_goal:
            self.current_best_goal = self.optim_status["goal"]
            self.current_best_params = self.optim_status["params"]
            if "U_dict" in self.exp.__dict__.keys():
                self.log_best_unitary()
            with open(self.logdir + "best_point_" + self.logname, "w") as best_point:
                best_point.write(json.dumps(self.opt_map))
                best_point.write("\n")
                best_point.write(json.dumps(self.optim_status))
                best_point.write("\n")
                best_point.write(self.nice_print(self.opt_map))
        if self.plot_dynamics:
            psi_init = self.exp.model.tasks["init_ground"].initialise(
                self.exp.model.drift_H, self.exp.model.lindbladian
            )
            # dim = np.prod(self.exp.model.dims)
            # psi_init = [0] * dim
            # psi_init[1] = 1
            # psi_init = tf.constant(psi_init, dtype=tf.complex128, shape=[dim ,1])
            for gate in self.exp.dUs.keys():
                self.exp.plot_dynamics(psi_init, [gate], self.optim_status["goal"])
            self.exp.dynamics_plot_counter += 1
        if self.plot_pulses:
            for gate in self.opt_gates:
                instr = self.exp.gateset.instructions[gate]
                self.exp.plot_pulses(instr, self.optim_status["goal"])
            self.exp.pulses_plot_counter += 1
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

    def fct_to_min(self, x):
        """
        Wrapper for the goal function.

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
        goal = self.goal_run(current_params)
        self.log_parameters()
        if isinstance(goal, tf.Tensor):
            goal = float(goal.numpy())
        return goal

    def fct_to_min_autograd(self, x):
        """
         Wrapper for the goal function, including evaluation and storage of the gradient.

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
            goal = float(goal.numpy())
        return goal

    def goal_run_with_grad(self, current_params):
        """OBSOLETE?"""
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

    def write_config(self, filename):
        with open(filename, "w") as cfg_file:
            json.dump(self.__dict__, cfg_file)

    def load_config(self, filename):
        with open(filename, "r") as cfg_file:
            cfg = json.loads(cfg_file.read(1))
        for key in cfg:
            if key == "gateset":
                self.gateset.load_config(cfg[key])
            elif key == "sim":
                self.sim.load_config(cfg[key])
            elif key == "exp":
                self.exp.load_config(cfg[key])
            else:
                self.__dict__[key] = cfg[key]

    # TODO fix error when JSONing fucntion types
