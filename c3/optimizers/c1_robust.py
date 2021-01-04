import os
import time
import json
import tensorflow as tf
import c3.utils.display as display
from c3.optimizers.optimizer import Optimizer
from c3.utils.utils import log_setup
import matplotlib.pyplot as plt
from c3.optimizers.c1 import C1
import copy
import numpy as np


class C1_robust(C1):
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
    gateset_opt_map : list
        Hierarchical identifiers for the parameter vector
    opt_gates : list
        List of identifiers of gate to be optimized, a subset of the full gateset
    callback_fids : list of callable
        Additional fidelity function to be evaluated and stored for reference
    algorithm : callable
        From the algorithm library
    plot_dynamics : boolean
        Save plots of time-resolved dynamics in dir_path
    plot_pulses : boolean
        Save plots of control signals
    store_unitaries : boolean
        Store propagators as text and pickle
    options : dict
        Options to be passed to the algorithm
    run_name : str
        User specified name for the run, will be used as root folder
    """

    def __init__(
            self,
            dir_path,
            fid_func,
            fid_subspace,
            gateset_opt_map,
            noise_map,
            opt_gates,
            callback_fids=[],
            algorithm=None,
            plot_dynamics=False,
            plot_pulses=False,
            store_unitaries=False,
            options={},
            run_name=None,
            interactive=True,
            num_runs=1
    ):
        super().__init__(
            dir_path=dir_path,
            fid_func=fid_func,
            fid_subspace=fid_subspace,
            gateset_opt_map=gateset_opt_map,
            opt_gates=opt_gates,
            callback_fids=callback_fids,
            algorithm=algorithm,
            plot_dynamics=plot_dynamics,
            plot_pulses=plot_pulses,
            store_unitaries=store_unitaries,
            options=options,
            run_name=run_name,
            interactive=interactive
        )
        self.num_runs = num_runs
        self.noise_map = noise_map

    # def goal_run(self, current_params):
    #     """
    #     Evaluate the goal function for current parameters.
    #
    #     Parameters
    #     ----------
    #     current_params : tf.Tensor
    #         Vector representing the current parameter values.
    #
    #     Returns
    #     -------
    #     tf.float64
    #         Value of the goal function
    #     """
    #     self.exp.gateset.set_parameters(
    #         current_params,
    #         self.opt_map,
    #         scaled=True
    #     )
    #     dims = self.exp.model.dims
    #     avg_goal = 0
    #     goals = []
    #
    #     # @tf.function
    #     def get_goal(i):
    #         exp = self.exp
    #         U_dict = exp.get_gates()
    #         try:
    #             goal = self.fid_func(U_dict, self.index, dims, self.evaluation + 1)
    #         except TypeError:
    #             goal = self.fid_func(exp, U_dict, self.index, dims, self.evaluation + 1)
    #         return goal
    #
    #     goals = tf.map_fn(get_goal, tf.range(self.num_runs, dtype=tf.float64))
    #     goal = tf.reduce_mean(goals)
    #     std = tf.math.reduce_std(goals)
    #     #
    #     #     avg_goal += goal
    #     #     goals.append((float(goal)))
    #     # goal = avg_goal / self.num_runs
    #
    #     try:
    #         display.plot_C1(self.logdir, interactive=self.interactive)
    #     except TypeError:
    #         pass
    #
    #     with open(self.logdir + self.logname, 'a') as logfile:
    #         logfile.write(f"\nEvaluation {self.evaluation + 1} returned:\n")
    #         logfile.write(
    #             "goal: {}: {}, std:{} iterations:{}\n".format(self.fid_func.__name__, float(goal), float(std), goals.numpy().tolist())
    #         )
    #         for cal in self.callback_fids:
    #             val = cal(
    #                 U_dict, self.index, dims, self.evaluation + 1
    #             )
    #             if isinstance(val, tf.Tensor):
    #                 val = float(val.numpy())
    #             logfile.write("{}: {}\n".format(cal.__name__, val))
    #             self.optim_status[cal.__name__] = val
    #         logfile.flush()
    #
    #     self.optim_status['params'] = [
    #         par.numpy().tolist()
    #         for par in self.exp.gateset.get_parameters(self.opt_map)
    #     ]
    #     self.optim_status['goal'] = float(goal)
    #     self.optim_status['time'] = time.asctime()
    #     self.evaluation += 1
    #
    #     return goal

    def goal_run_with_grad(self, current_params):
        """OBSOLETE?"""
        goals = []
        goals_float = []
        grads = []
        evaluation = int(self.evaluation)
        fig = plt.figure()
        for noise_vals, noise_map in self.noise_map:
            for noise_val in noise_vals:
                self.exp.set_parameters([noise_val], noise_map)
                self.evaluation = evaluation
                with tf.GradientTape() as t:
                    t.watch(current_params)
                    goal = self.goal_run(current_params)
                grad = t.gradient(goal, current_params)
                goals.append(goal)
                goals_float.append(float(goal))
                grads.append(grad)
            plt.plot(noise_vals, goals, 'x', label=str(noise_map))
            self.exp.set_parameters([0], noise_map)
        plt.legend()
        plt.xlabel('Robust_value')
        plt.ylabel('infidelity')
        plt.yscale('log')
        plt.savefig(self.logdir + 'robustness/' + f'eval_{evaluation + 1}_{float(tf.reduce_mean(goals))}.png', dpi=300)
        plt.cla()
        plt.clf()

        with open(self.logdir + self.logname, 'a') as logfile:
            logfile.write(f"\n------------------------\n")
            logfile.write(f"\nTotal Evaluation {evaluation + 1} returned:\n")
            logfile.write(
                "goal: {}: {}, std:{}, individual goals:{} \nstd_grad: {}\n".format(self.fid_func.__name__, float(tf.math.reduce_mean(goals)), float(tf.math.reduce_std(goals)), goals_float, (tf.math.reduce_std(grads, axis=0).numpy().tolist()))
            )
            logfile.flush()

        self.optim_status['goal'] = float(tf.reduce_mean(goals, axis=0))
        self.optim_status['time'] = time.asctime()
        return tf.reduce_mean(goals, axis=0), tf.reduce_mean(grads, axis=0)

    def jsonify_list(self, data):
        if isinstance(data, dict):
            return {k: self.jsonify_list(v) for k, v in data.items()}
        elif isinstance(data, list):
            return [self.jsonify_list(v) for v in data]
        elif isinstance(data, np.ndarray):
            return data.tolist()
        # elif isinstance(data, tf.eaget)
        else:
            return data

    def start_log(self):
        """
        Initialize the log with current time.

        """
        super().start_log()
        with open(self.logdir + self.logname, 'a') as logfile:
            logfile.write("Robust values ")

            logfile.write(json.dumps(self.jsonify_list(self.noise_map)))
            logfile.write("\n")
            logfile.flush()
        os.mkdir(self.logdir + 'robustness')