"""Object that deals with the open loop optimal control."""

import os
import json
import c3po.display
from c3po.optimizer import Optimizer
import matplotlib.pyplot as plt
from c3po.utils import log_setup

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
            algorithm_no_grad=algorithm_no_grad,
            algorithm_with_grad=algorithm_with_grad
            )
        self.opt_map = gateset_opt_map
        self.fid_func = fid_func
        self.callback_fids = callback_fids
        self.optim_status = {}
        self.gradients = {}
        self.evaluation = 1
        self.log_setup(dir_path)

    def log_setup(self, dir_path):
        string = self.fid_func.__name__
        self.logdir = log_setup(dir_path, string)
        self.logfile_name = self.logdir + 'open_loop.log'

    def optimize_controls(self):
        """
        Apply a search algorightm to your gateset given a fidelity function.
        """
        self.start_log()

        print(f"Saving as:\n{self.logfile_name}")
        os.makedirs(self.logdir + 'dynamics_xyxy')
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
        goal = self.eval_func(U_dict)

        fig, axs = self.exp.plot_dynamics(
            self.exp.psi_init,
            ['X90p', 'Y90p', 'X90p', 'Y90p']
        )
        fig.savefig(
            self.logdir +
            + 'dynamics_xyxy/' +
            + 'eval:' + str(self.evaluation) + "__" +
            + self.fom.__name__ + str(round(goal.numpy(), 3)) +
            + '.png'
        )
        plt.close(fig)
        c3po.display.plot_OC_logs(self.logdir)

        self.optim_status['params'] = [
            par.numpy().tolist()
            for par in self.exp.get_parameters(self.opt_map)
        ]
        self.optim_status['goal'] = float(goal.numpy())
        return goal
