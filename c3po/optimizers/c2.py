"""Object that deals with the closed loop optimal control."""

import json
import pickle
from c3po.optimizers.optimizer import Optimizer
from c3po.utils.utils import log_setup


class C2(Optimizer):
    """Object that deals with the closed loop optimal control."""

    def __init__(
        self,
        dir_path,
        eval_func,
        gateset_opt_map,
        algorithm_no_grad
    ):
        """Initiliase."""
        super().__init__(
            algorithm_no_grad=algorithm_no_grad,
            algorithm_with_grad=None
            )
        self.eval_func = eval_func
        self.opt_map = gateset_opt_map
        self.log_setup(dir_path)

    def log_setup(self, dir_path):
        string = self.eval_func.__name__ + self.algorithm.__name__
        self.logdir = log_setup(dir_path, string)
        self.logfile_name = self.logdir + 'closed_loop.log'

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
        self.picklefilename = self.logdir + "learn_from.pickle"
        print(f"Saving as:\n{self.logfile_name}")
        self.nice_print = self.exp.gateset.print_parameters
        x0 = self.exp.gateset.get_parameters(self.opt_map, scaled=True)
        try:
            self.algorithm(
                x0,
                self.fct_to_min
            )
        except KeyboardInterrupt:
            pass
        self.end_log()
        with open(self.picklefilename, "rb") as file:
            measurements = pickle.load(file)
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
        return goal

        def log_pickle(self, params, seqs, results, results_std):
            m = {}
            m['params'] = params
            m['seqs'] = seqs
            m['results'] = results
            m['result_stds'] = results_std
            with open(self.picklefilename, "ab+") as file:
                pickle.dump(m, file)
