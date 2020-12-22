"""Object that deals with the closed loop optimal control."""

import os
import shutil
import time
import hjson
import pickle
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
        exp_right=None,
        options={},
        run_name=None,
    ):
        super().__init__(pmap=pmap, algorithm=algorithm)
        self.eval_func = eval_func
        self.options = options
        self.exp_right = exp_right
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

    def log_setup(self) -> None:
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
        self.logname = "calibration.log"
        shutil.copy2(self.eval_func, self.logdir)
        real_log = os.path.join(self.logdir, "real_model.hjson")
        self.exp_right.pmap.model.write_config(real_log)

    def optimize_controls(self) -> None:
        """
        Apply a search algorithm to your gateset given a fidelity function.
        """
        self.log_setup()
        self.start_log()
        self.picklefilename = self.logdir + "dataset.pickle"
        print(f"C3:STATUS:Saving as: {os.path.abspath(self.logdir + self.logname)}")
        x_init = self.pmap.get_parameters_scaled()
        try:
            self.algorithm(
                x_init,
                fun=self.fct_to_min,
                fun_grad=self.fct_to_min_autograd,
                grad_lookup=self.lookup_gradient,
                options=self.options,
            )
        except KeyboardInterrupt:
            pass
        with open(self.logdir + "best_point_" + self.logname, "r") as file:
            best_params = hjson.loads(file.readlines()[1])["params"]
        self.pmap.set_parameters(best_params)
        self.end_log()
        measurements = []
        with open(self.picklefilename, "rb") as pickle_file:
            while True:
                try:
                    measurements.append(pickle.load(pickle_file))
                except EOFError:
                    break
        learn_from = {}
        learn_from["seqs_grouped_by_param_set"] = measurements
        learn_from["opt_map"] = self.pmap.opt_map
        with open(self.picklefilename, "wb+") as pickle_file:
            pickle.dump(learn_from, pickle_file)

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
        self.optim_status["params"] = [
            par.numpy().tolist() for par in self.pmap.get_parameters()
        ]
        self.optim_status["goal"] = float(goal)
        self.optim_status["time"] = time.asctime()
        self.evaluation += 1
        self.log_pickle(params, seqs, results, results_std, shots)
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
        data_entry = {
            "params": params,
            "seqs": seqs,
            "results": results,
            "results_std": results_std,
            "shots": shots,
        }
        with open(self.picklefilename, "ab") as file:
            pickle.dump(data_entry, file)
