"""Object that deals with the open loop optimal control."""

import os
import json
import tensorflow as tf
import c3po.utils.display as display
from c3po.optimizers.optimizer import Optimizer
from c3po.utils.utils import log_setup


class C1(Optimizer):
    """Object that deals with the open loop optimal control."""

    def __init__(
        self,
        dir_path,
        fid_func,
        fid_subspace,
        gateset_opt_map,
        opt_gates,
        callback_fids=[],
        algorithm_no_grad=None,
        algorithm_with_grad=None,
        plot_dynamics=False,
        plot_pulses=False,
        options={}
    ):
        """Initiliase."""
        super().__init__(
            algorithm_no_grad=algorithm_no_grad,
            algorithm_with_grad=algorithm_with_grad,
            plot_dynamics=plot_dynamics,
            plot_pulses=plot_pulses
            )
        self.opt_map = gateset_opt_map
        self.opt_gates = opt_gates
        self.fid_func = fid_func
        self.fid_subspace = fid_subspace
        self.callback_fids = callback_fids
        self.options = options
        self.log_setup(dir_path)

    def log_setup(self, dir_path):
        self.dir_path = dir_path
        self.string = (
            'c1_' + self.fid_func.__name__ + '_' + self.algorithm.__name__
        )
        self.logdir = log_setup(dir_path, self.string)
        self.logname = 'open_loop.log'

    def load_best(self, init_point):
        with open(init_point) as init_file:
            best = init_file.readlines()
            best_gateset_opt_map = [
                [tuple(par) for par in set]
                for set in json.loads(best[0])
            ]
            init_p = json.loads(best[1])['params']
            self.exp.gateset.set_parameters(init_p, best_gateset_opt_map)

    def optimize_controls(self):
        """
        Apply a search algorightm to your gateset given a fidelity function.
        """
        self.start_log()
        self.exp.set_enable_dynamics_plots(self.plot_dynamics, self.logdir)
        self.exp.set_enable_pules_plots(self.plot_pulses, self.logdir)
        self.exp.set_opt_gates(self.opt_gates)
        self.nice_print = self.exp.gateset.print_parameters
        print(f"Saving as:    {os.path.abspath(self.logdir + self.logname)}")
        index = []
        for name in self.fid_subspace:
            index.append(self.exp.model.names.index(name))
        self.index = index
        x0 = self.exp.gateset.get_parameters(self.opt_map, scaled=True)
        try:
            # TODO deal with kears learning differently
            if self.grad:
                self.algorithm(
                    x0,
                    self.fct_to_min,
                    self.lookup_gradient,
                    self.options
                )
            else:
                self.algorithm(
                    x0,
                    self.fct_to_min,
                    self.options
                )
        except KeyboardInterrupt:
            pass
        with open(self.logdir + 'best_point_' + self.logname, 'r') as file:
            best_params = json.loads(file.readlines()[1])['params']
        self.exp.gateset.set_parameters(best_params, self.opt_map)
        self.end_log()

    def goal_run(self, current_params):
        self.exp.gateset.set_parameters(
            current_params,
            self.opt_map,
            scaled=True
        )
        dims = self.exp.model.dims
        U_dict = self.exp.get_gates()
        goal = self.fid_func(U_dict, self.index, dims)
        goal_numpy = float(goal.numpy())
        display.plot_C1(self.logdir)

        with open(self.logdir + self.logname, 'a') as logfile:
            logfile.write(f"\nEvaluation {self.evaluation + 1} returned:\n")
            logfile.write(
                "goal: {}: {}\n".format(self.fid_func.__name__, goal_numpy)
            )
            for cal in self.callback_fids:
                val = cal(U_dict)
                if isinstance(val, tf.Tensor):
                    val = float(val.numpy())
                logfile.write("{}: {}\n".format(cal.__name__, val))
                self.optim_status[cal.__name__] = val
            logfile.flush()

        self.optim_status['params'] = [
            par.numpy().tolist()
            for par in self.exp.gateset.get_parameters(self.opt_map)
        ]
        self.optim_status['goal'] = goal_numpy
        self.evaluation += 1
        return goal
