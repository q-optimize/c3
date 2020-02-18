"""Optimizer object, where the optimal control is done."""

import time
import json
import tensorflow as tf
import c3po.libraries.algorithms


class Optimizer:
    """Optimizer object, where the optimal control is done."""

    def __init__(
        self,
        algorithm_no_grad=None,
        algorithm_with_grad=None,
    ):
        self.optim_status = {}
        self.gradients = {}
        self.current_best_goal = 9876543210.123456789
        self.evaluation = 0
        if algorithm_with_grad:
            self.algorithm = algorithm_with_grad
            self.grad = True
        elif algorithm_no_grad:
            self.algorithm = algorithm_no_grad
            self.grad = False
        else:
            raise Exception("No algorithm passed. Using default LBFGS")
            self.algorithm = c3po.algorithms.lbfgs
            self.grad = True

    def set_exp(self, exp):
        self.exp = exp

    def start_log(self):
        self.start_time = time.time()
        start_time_str = str(f"{time.asctime(time.localtime())}\n\n")
        with open(self.logfile_name, 'a') as logfile:
            logfile.write("Starting optimization at ")
            logfile.write(start_time_str)
            logfile.write("Optimization parameters:\n")
            logfile.write(json.dumps(self.opt_map))
            logfile.write("\n")
            logfile.flush()

    def end_log(self):
        self.end_time = time.time()
        with open(self.logfile_name, 'a') as logfile:
            logfile.write(
                f"Finished at {time.asctime(time.localtime())}\n"
            )
            logfile.write(
                f"Total runtime: {self.end_time-self.start_time}\n\n"
            )
            logfile.flush()

    def log_parameters(self):
        if self.optim_status['goal'] < self.current_best_goal:
            self.current_best_goal = self.optim_status['goal']
            with open(self.logdir+'best_point', 'w') as best_point:
                best_point.write(json.dumps(self.opt_map))
                best_point.write("\n")
                best_point.write(json.dumps(self.optim_status))
                best_point.write("\n")
                best_point.write(self.nice_print(self.opt_map))
        with open(self.logfile_name, 'a') as logfile:
            logfile.write(json.dumps(self.optim_status))
            logfile.write("\n")
            logfile.write(f"\nFinished evaluation {self.evaluation}\n")
            logfile.flush()

    def fct_to_min(self, x):
        current_params = tf.constant(x)
        if self.grad:
            goal = self.goal_run_with_grad(current_params)
        else:
            goal = self.goal_run(current_params)
        self.log_parameters()
        return float(goal.numpy())

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
