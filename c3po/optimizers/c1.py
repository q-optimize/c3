"""Object that deals with the model learning."""

import time
import json
import pickle
import numpy as np
import tensorflow as tf
from c3po.optimizers.optimizer import Optimizer
import matplotlib.pyplot as plt

class C1(Optimizer):
    """Object that deals with the open loop optimal control."""
    def __init__(
        self,
        dir_path,
        gateset_opt_map,
        fid_func,
        callback_fids=[],
        algorithm_no_grad=None,
        algorithm_with_grad=None,
    ):
        """Initiliase."""
        super().__init__(
            dir_path=dir_path,
            algorithm_no_grad=algorithm_no_grad,
            algorithm_with_grad=algorithm_with_grad
            )
        self.opt_map = gateset_opt_map
        self.fid_func = fid_func
        self.callback_fids = callback_fids
        super().__init__()

    def goal_run(self, current_params):
        self.gateset.set_parameters(current_params, self.opt_map, scaled=True)
        U_dict = self.sim.get_gates()
        goal = self.eval_func(U_dict)
        self.optim_status['params'] = [
            par.numpy().tolist() for par in self.exp.get_parameters(self.opt_map)
        ]
        self.optim_status['goal'] = float(goal.numpy())
        self.log_parameters()
        return goal

    def goal_run_with_grad(self, current_params):
        with tf.GradientTape() as t:
            t.watch(current_params)
            self.gateset.set_parameters(
                current_params, self.opt_map, scaled=True
            )
            U_dict = self.sim.get_gates()
            goal = self.eval_func(U_dict)

        grad = t.gradient(goal, current_params)
        gradients = grad.numpy().flatten()
        self.gradients[str(current_params.numpy())] = gradients
        self.optim_status['params'] = [
            par.numpy().tolist() for par in self.exp.get_parameters(self.opt_map)
        ]
        self.optim_status['goal'] = float(goal.numpy())
        self.optim_status['gradient'] = gradients.tolist()
        self.log_parameters()
        return goal


    def optimize_controls(
        self,
        sim,
        gateset,
        opt_map,
        opt_name,
        fid_func,
        callbacks=[],
        settings={},
        other_funcs={}
    ):
        """
        Apply a search algorightm to your gateset given a fidelity function.

        Parameters
        ----------
        simulator : class Simulator
            simulator class carrying all relevant informatioFn:
            - experiment object with model and generator
            - gateset object with pulse information for each gate

        opt_map : list
            Specifies which parameters will be optimized

        algorithm : str
            Specification of the optimizer to be used, i.e. cmaes, powell, ...

        calib_name : str

        eval_func : function
            Takes the dictionary of gates and outputs a fidelity value of which
            we want to find the minimum

        settings : dict
            Special settings for the desired optimizer

        other_funcs : dict of functions
            All functions that will be calculated from U_dict and stored

        """
        # TODO Separate gateset from the simulation here.
        x0 = gateset.get_parameters(opt_map, scaled=True)
        self.init_values = x0
        self.sim = sim
        self.gateset = gateset
        self.opt_map = opt_map
        self.opt_name = opt_name
        self.fid_func = fid_func
        self.callbacks = callbacks
        self.optim_status = {}
        self.evaluation = 1

        # TODO log physical values, not tf values

        self.logfile_name = self.data_path + self.opt_name + '.log'
        print(f"Saving as:\n{self.logfile_name}")
        start_time = time.time()
        with open(self.logfile_name, 'a') as self.logfile:
            start_time_str = str(f"{time.asctime(time.localtime())}\n\n")
            self.logfile.write("Starting optimization at")
            self.logfile.write(start_time_str)
            self.logfile.write(f"\n {self.opt_map}\n")
            self.logfile.flush()
            if self.algorithm == 'cmaes':
                self.cmaes(
                    x0,
                    self.goal_run,
                    settings
                )

            elif self.algorithm == 'lbfgs':
                x_best = self.lbfgs(
                    x0,
                    self.goal_run_with_grad,
                    options=settings
                )

            self.gateset.set_parameters(
                x_best, self.opt_map, scaled=True
            )
            end_time = time.time()
            self.logfile.write("Started at ")
            self.logfile.write(start_time_str)
            self.logfile.write(
                f"Finished at {time.asctime(time.localtime())}\n"
            )
            self.logfile.write(
                f"Total runtime:{end_time-start_time}\n\n"
            )
            self.logfile.flush()

        # TODO decide if gateset object should have history and implement
        # TODO save while setting if you pass a save name
        # pseudocode: controls.save_params_to_history(calib_name)
