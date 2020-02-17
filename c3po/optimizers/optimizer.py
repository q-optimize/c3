"""Optimizer object, where the optimal control is done."""

import time
import json
import c3po.libraries.algorithms

class Optimizer:
    """Optimizer object, where the optimal control is done."""

    def __init__(
        self,
        algorithm_no_grad=None,
        algorithm_with_grad=None,
    ):
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

    def log_parameters(self):
        if self.optim_status['goal'] < self.current_best_goal:
            self.current_best_goal = self.optim_status['goal']
            with open(self.data_path+'best_point', 'w') as best_point:
                best_point.write(json.dumps(self.opt_map))
                best_point.write("\n")
                best_point.write(json.dumps(self.optim_status))
        self.logfile.write(json.dumps(self.optim_status))
        self.logfile.write("\n")
        self.logfile.write(f"\nFinished evaluation {self.evaluation}\n")
        self.evaluation += 1
        self.logfile.flush()

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


    def lookup_gradient(self, x):
        key = str(x)
        return self.gradients.pop(key)

    # TODO fix error when JSONing fucntion types
