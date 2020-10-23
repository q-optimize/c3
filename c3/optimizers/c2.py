"""Object that deals with the closed loop optimal control."""

import os
import time
import json
import pickle
import c3.utils.display as display
from c3.optimizers.optimizer import Optimizer
from c3.utils.utils import log_setup


class C2(Optimizer):
    """
    Object that deals with the closed loop optimal control.

    Parameters
    ----------
    dir_path : str
        Filepath to save results
    eval_func : callable
        infidelity function to be minimized
    pmap : ParameterMap
        Identifiers for the parameter vector
    algorithm : callable
        From the algorithm library
    options : dict
        Options to be passed to the algorithm
    run_name : str
        User specified name for the run, will be used as root folder
    """

    def __init__(
        self,
        dir_path,
        eval_func,
        pmap,
        algorithm,
        options={},
        run_name=None,
    ):
        super().__init__(
            algorithm=algorithm
            )
        self.eval_func = eval_func
        # TODO the pmap can go to the optimizer super class
        self.pmap = pmap
        self.options = options
        self.__dir_path = dir_path
        self.__run_name = run_name

    def set_eval_func(self, eval_func):
        """
        Setter for the eval function.

        Parameters
        ----------
        eval_func : callable
            Function to be evaluated

        """
        self.eval_func = eval_func

    def log_setup(self):
        """
        Create the folders to store data.

        Parameters
        ----------
        dir_path : str
            Filepath
        run_name : str
            User specified name for the run

        """
        dir_path = os.path.abspath(self.__dir_path)
        run_name = self.__run_name
        if run_name is None:
            run_name = self.eval_func.__name__ + self.algorithm.__name__
        self.logdir = log_setup(dir_path, run_name)
        self.logname = 'calibration.log'

    def load_best(self, init_point):
        """
        Load a previous parameter point to start the optimization from.

        Parameters
        ----------
        init_point : str
            File location of the initial point

        """
        with open(init_point) as init_file:
            best = init_file.readlines()
            best_gateset_opt_map = [
                [tuple(par) for par in set]
                for set in json.loads(best[0])
            ]
            init_p = json.loads(best[1])['params']
            self.pmap.set_parameters(init_p, best_gateset_opt_map)

    def optimize_controls(self):
        """
        Apply a search algorithm to your gateset given a fidelity function.
        """
        self.log_setup()
        self.start_log()
        self.picklefilename = self.logdir + "dataset.pickle"
        print(f"C3:STATUS:Saving as: {os.path.abspath(self.logdir + self.logname)}")
        x0 = self.pmap.get_parameters_scaled()
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
        self.pmap.set_parameters(best_params)
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
        learn_from['opt_map'] = self.pmap.opt_map
        with open(self.picklefilename, "wb+") as file:
            pickle.dump(learn_from, file)

    def goal_run(self, current_params):
        """
        Evaluate the goal function for current parameters.

        Parameters
        ----------
        current_params : tf.Tensor
            Vector representing the current parameter values.

        Returns
        -------
        tf.float64
            Value of the goal function
        """
        self.pmap.set_parameters_scaled(current_params)
        # There could be more processing happening here, i.e. an exp could
        # generate signals for an experiment and send those to eval_func.
        params = self.pmap.get_parameters()

        goal, results, results_std, seqs, shots = self.eval_func(params)
        self.optim_status['params'] = [
            par.numpy().tolist()
            for par in self.pmap.get_parameters()
        ]
        self.optim_status['goal'] = float(goal)
        self.optim_status['time'] = time.asctime()
        self.evaluation += 1
        self.log_pickle(params, seqs, results, results_std, shots)
        display.plot_C2(self.__dir_path, self.logdir)
        return goal

    def log_pickle(self, params, seqs, results, results_std, shots):
        """
        Save a pickled version of the performed experiment, suitable for model learning.

        Parameters
        ----------
        params : tf.Tensor
            Vector of parameter values
        seqs : list
            Strings identifying the performed instructions
        results : list
            Values of the goal function
        results_std : list
            Standard deviation of the results, in the case of noisy data
        shots : list
            Number of repetitions used in averaging noisy data

        """
        m = {'params': params, 'seqs': seqs, 'results': results, 'results_std': results_std, 'shots': shots}
        with open(self.picklefilename, "ab") as file:
            pickle.dump(m, file)
