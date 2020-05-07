"""Optimizer object, where the optimal control is done."""

import os
import time
import json
import tensorflow as tf
import numpy as np
import c3po.libraries.algorithms as algorithms


class Optimizer:
    """Optimizer object, where the optimal control is done."""

    def __init__(
        self,
        algorithm=None,
        plot_dynamics=False,
        plot_pulses=False
    ):
        self.optim_status = {}
        self.gradients = {}
        self.current_best_goal = 9876543210.123456789
        self.evaluation = 0
        self.plot_dynamics = plot_dynamics
        self.plot_pulses = plot_pulses
        if algorithm is not None:
            self.algorithm = algorithm
        else:
            print("C3:WARNING:No algorithm passed. Using default LBFGS")
            self.algorithm = algorithms.lbfgs

    def replace_logdir(self, new_logdir):
        old_logdir = self.logdir
        self.logdir = new_logdir
        os.remove(self.dir_path + 'recent')
        os.remove(self.dir_path + self.string)
        os.rmdir(old_logdir)

    def set_exp(self, exp):
        self.exp = exp

    def start_log(self):
        self.start_time = time.time()
        start_time_str = str(f"{time.asctime(time.localtime())}\n\n")
        with open(self.logdir + self.logname, 'a') as logfile:
            logfile.write("Starting optimization at ")
            logfile.write(start_time_str)
            logfile.write("Optimization parameters:\n")
            logfile.write(json.dumps(self.opt_map))
            logfile.write("\n")
            logfile.flush()

    def end_log(self):
        self.end_time = time.time()
        with open(self.logdir + self.logname, 'a') as logfile:
            logfile.write(
                f"Finished at {time.asctime(time.localtime())}\n"
            )
            logfile.write(
                f"Total runtime: {self.end_time-self.start_time}\n\n"
            )
            logfile.flush()

    def log_best_unitary(self):
        if self.optim_status['goal'] < self.current_best_goal:
            self.current_best_goal = self.optim_status['goal']
            with open(
                self.logdir + 'best_point_' + self.logname, 'w'
            ) as best_point:
                U_dict = self.exp.unitaries
                for gate, U in U_dict.items():
                    best_point.write("\n")
                    best_point.write(f"Re {gate}: \n")
                    best_point.write(f"{np.round(np.real(U), 3)}\n")
                    best_point.write("\n")
                    best_point.write(f"Im {gate}: \n")
                    best_point.write(f"{np.round(np.imag(U), 3)}\n")

    def log_parameters(self):
        if self.optim_status['goal'] < self.current_best_goal:
            self.current_best_goal = self.optim_status['goal']
            with open(
                self.logdir + 'best_point_' + self.logname, 'w'
            ) as best_point:
                best_point.write(json.dumps(self.opt_map))
                best_point.write("\n")
                best_point.write(json.dumps(self.optim_status))
                best_point.write("\n")
                best_point.write(self.nice_print(self.opt_map))
            if self.plot_dynamics:
                psi_init = self.exp.model.tasks["init_ground"].initialise(
                    self.exp.model.drift_H,
                    self.exp.model.lindbladian
                )
                for gate in self.exp.dUs.keys():
                    self.exp.plot_dynamics(psi_init, [gate])
                self.exp.dynamics_plot_counter += 1
        if self.plot_pulses:
            psi_init = self.exp.model.tasks["init_ground"].initialise(
                self.exp.model.drift_H,
                self.exp.model.lindbladian
            )
            for gate in self.exp.gateset.instructions.keys():
                instr = self.exp.gateset.instructions[gate]
                self.exp.plot_pulses(instr)
            self.exp.pulses_plot_counter += 1
        with open(self.logdir + self.logname, 'a') as logfile:
            logfile.write(f"\nFinished evaluation {self.evaluation}\n")
            # logfile.write(json.dumps(self.optim_status, indent=2))
            logfile.write(json.dumps(self.optim_status))
            logfile.write("\n")
            logfile.flush()

    def fct_to_min(self, x):
        current_params = tf.constant(x)
        goal = self.goal_run(current_params)
        self.log_parameters()
        if "U_dict" in self.exp.__dict__.keys():
            self.log_best_unitary()
        if isinstance(goal, tf.Tensor):
            goal = float(goal.numpy())
        return goal

    def fct_to_min_autograd(self, x):
        current_params = tf.constant(x)
        goal = self.goal_run_with_grad(current_params)
        self.log_parameters()
        if "U_dict" in self.exp.__dict__.keys():
            self.log_best_unitary()
        if isinstance(goal, tf.Tensor):
            goal = float(goal.numpy())
        return goal

    def goal_run_with_grad(self, current_params):
        with tf.GradientTape() as t:
            t.watch(current_params)
            goal = self.goal_run(current_params)
        grad = t.gradient(goal, current_params)
        gradients = grad.numpy().flatten()
        self.gradients[str(current_params.numpy())] = gradients
        self.optim_status['gradient'] = gradients.tolist()
        return goal

    def lookup_gradient(self, x):
        key = str(x)
        return self.gradients.pop(key)

    def write_config(self, filename):
        with open(filename, "w") as cfg_file:
            json.dump(self.__dict__, cfg_file)

    def load_config(self, filename):
        with open(filename, "r") as cfg_file:
            cfg = json.loads(cfg_file.read(1))
        for key in cfg:
            if key == 'gateset':
                self.gateset.load_config(cfg[key])
            elif key == 'sim':
                self.sim.load_config(cfg[key])
            elif key == 'exp':
                self.exp.load_config(cfg[key])
            else:
                self.__dict__[key] = cfg[key]

    # TODO fix error when JSONing fucntion types
