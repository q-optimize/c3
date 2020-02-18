"""Object that deals with the open loop optimal control."""

import json
import c3po.utils.display as display
from c3po.optimizers.optimizer import Optimizer
from c3po.utils.utils import log_setup


class C1(Optimizer):
    """Object that deals with the open loop optimal control."""

    def __init__(
        self,
        dir_path,
        fid_func,
        gateset_opt_map,
        callback_fids=[],
        algorithm_no_grad=None,
        algorithm_with_grad=None,
    ):
        """Initiliase."""
        super().__init__(
            algorithm_no_grad=algorithm_no_grad,
            algorithm_with_grad=algorithm_with_grad
            )
        self.opt_map = gateset_opt_map
        self.fid_func = fid_func
        self.callback_fids = callback_fids
        self.log_setup(dir_path)

    def log_setup(self, dir_path):
        string = self.fid_func.__name__ + self.algorithm.__name__
        self.logdir = log_setup(dir_path, string)
        self.logfile_name = self.logdir + 'open_loop.log'

    def load_best(self, init_point):
        with open(init_point) as init_file:
            best = init_file.readlines()
            best_gateset_opt_map = [tuple(a) for a in json.loads(best[0])]
            init_p = json.loads(best[1])['params']
            self.exp.gateset.set_parameters(init_p, best_gateset_opt_map)
            print("Loading previous best point.")

    def optimize_controls(self):
        """
        Apply a search algorightm to your gateset given a fidelity function.
        """
        self.start_log()

        print(f"Saving as:\n{self.logfile_name}")
        x0 = self.exp.gateset.get_parameters(self.opt_map, scaled=True)
        try:
            # TODO deal with kears learning differently
            if self.grad:
                self.algorithm(
                    x0,
                    self.fct_to_min,
                    self.lookup_gradient
                )
            else:
                self.algorithm(
                    x0,
                    self.fct_to_min
                )
        except KeyboardInterrupt:
            pass
        with open(self.logdir + 'best_point', 'r') as file:
            best_params = json.loads(file.readlines()[1])['params']
        self.exp.gateset.set_parameters(best_params, self.opt_map)
        self.end_log()

    def goal_run(self, current_params):
        self.exp.gateset.set_parameters(
            current_params,
            self.opt_map,
            scaled=True
        )
        U_dict = self.exp.get_gates()
        goal = self.fid_func(U_dict)
        # display.plot_OC_logs(self.logdir)

        self.optim_status['params'] = [
            par.numpy().tolist()
            for par in self.exp.gateset.get_parameters(self.opt_map)
        ]
        self.optim_status['goal'] = float(goal.numpy())
        return goal
