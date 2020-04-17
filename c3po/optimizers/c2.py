"""Object that deals with the closed loop optimal control."""

import os
import json
import pickle
import c3po.utils.display as display
from c3po.optimizers.optimizer import Optimizer
from c3po.utils.utils import log_setup


class C2(Optimizer):
    """Object that deals with the closed loop optimal control."""

    def __init__(
        self,
        dir_path,
        eval_func,
        gateset_opt_map,
        algorithm,
        options={}
    ):
        """Initiliase."""
        super().__init__(
            algorithm=algorithm
            )
        self.eval_func = eval_func
        self.opt_map = gateset_opt_map
        self.options = options
        self.log_setup(dir_path)

    def set_eval_func(self, eval_func):
        self.eval_func = eval_func

    def log_setup(self, dir_path):
        self.dir_path = dir_path
        self.string = self.eval_func.__name__ + self.algorithm.__name__
        self.logdir = log_setup(dir_path, self.string)
        self.logname = 'calibration.log'

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
        Apply a search algorithm to your gateset given a fidelity function.
        """
        self.start_log()
        self.picklefilename = self.logdir + "learn_from.pickle"
        print(f"C3:STATUS:Saving as: {os.path.abspath(self.logdir + self.logname)}")
        self.nice_print = self.exp.gateset.print_parameters
        x0 = self.exp.gateset.get_parameters(self.opt_map, scaled=True)
        try:
            self.algorithm(
                x0,
                fun=self.fct_to_min,
                fun_grad=self.fct_to_min_autograd,
                grad_lookup=self.lookup_gradient,
                options=self.options
            )
        except KeyboardInterrupt:
            pass
        with open(self.logdir + 'best_point_' + self.logname, 'r') as file:
            best_params = json.loads(file.readlines()[1])['params']
        self.exp.gateset.set_parameters(best_params, self.opt_map)
        self.end_log()
        measurements = []
        with open(self.picklefilename, "rb") as file:
            while True:
                try:
                    measurements.append(pickle.load(file))
                except EOFError:
                    break
        learn_from = {}
        learn_from['seqs_grouped_by_param_set'] = measurements
        learn_from['opt_map'] = self.opt_map
        with open(self.picklefilename, "wb+") as file:
            pickle.dump(learn_from, file)

    def goal_run(self, current_params):
        self.exp.gateset.set_parameters(
            current_params,
            self.opt_map,
            scaled=True
        )
        # There could be more processing happening here, i.e. the exp could
        # generate signals for an experiment and send those to eval_func.
        params = self.exp.gateset.get_parameters(self.opt_map, scaled=False)

        goal, results, results_std, seqs = self.eval_func(params)
        self.optim_status['params'] = [
            par.numpy().tolist()
            for par in self.exp.gateset.get_parameters(self.opt_map)
        ]
        self.optim_status['goal'] = float(goal)
        self.evaluation += 1
        self.log_pickle(params, seqs, results, results_std)
        display.plot_C2(self.dir_path, self.logdir)
        return goal

    def log_pickle(self, params, seqs, results, results_std):
        m = {}
        m['params'] = params
        m['seqs'] = seqs
        m['results'] = results
        m['results_std'] = results_std
        with open(self.picklefilename, "ab") as file:
            pickle.dump(m, file)
