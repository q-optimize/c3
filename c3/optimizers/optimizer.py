"""Optimizer object, where the optimal control is done."""

import os
import time
from typing import Callable, Union, List, Dict, Any
import numpy as np
import tensorflow as tf
import hjson
import c3.libraries.algorithms as algorithms
from c3.c3objs import hjson_encode
from c3.experiment import Experiment
from c3.parametermap import ParameterMap
import copy
from tensorboard.plugins.hparams import api as hp
import warnings


class Optimizer:
    """
    General optimizer class from which specific classes are inherited.

    Parameters
    ----------
    algorithm : callable
        From the algorithm library
    store_unitaries : boolean
        Store propagators as text and pickle
    logger: List
        Logging classes
    """

    def __init__(
        self,
        pmap: ParameterMap,
        algorithm: Callable = None,
        store_unitaries: bool = False,
        logger: List = None,
    ):
        self.pmap = pmap
        self.optim_status: Dict[str, Any] = dict()
        self.gradients: Dict[str, np.ndarray] = {}
        self.current_best_goal = 9876543210.123456789
        self.current_best_params = None
        self.evaluation = 0
        self.store_unitaries = store_unitaries
        self.created_by = None
        self.logname: str = None
        self.options = None
        self.__dir_path: str = None
        self.logdir: str = None
        self.set_algorithm(algorithm)
        self.logger = []
        if logger is not None:
            self.logger = logger

    def set_algorithm(self, algorithm: Callable) -> None:
        if algorithm:
            self.algorithm = algorithm
        else:
            print("C3:WARNING:No algorithm passed. Using default LBFGS")
            self.algorithm = algorithms.lbfgs

    def replace_logdir(self, new_logdir):
        """
        Specify a new filepath to store the log.

        Parameters
        ----------
        new_logdir

        """
        new_logdir = new_logdir.replace(":", "_")
        old_logdir = self.logdir
        self.logdir = new_logdir

        if old_logdir is None:
            return

        try:
            os.remove(os.path.join(self.__dir_path, "recent"))
        except FileNotFoundError:
            pass

        try:
            os.rmdir(old_logdir)
        except OSError:
            pass

        for logger in self.logger:
            logger.set_logdir

    def set_exp(self, exp: Experiment) -> None:
        self.exp = exp

    def set_created_by(self, config) -> None:
        """
        Store the config file location used to created this optimizer.
        """
        self.created_by = config

    def load_best(self, init_point) -> None:
        """
        Load a previous parameter point to start the optimization from. Legacy wrapper.
        Method moved to Parametermap.

        Parameters
        ----------
        init_point : str
            File location of the initial point

        """
        self.pmap.load_values(init_point)

    def start_log(self) -> None:
        """
        Initialize the log with current time.

        """
        self.start_time = time.time()
        start_time_str = str(f"{time.asctime(time.localtime())}\n\n")
        with open(self.logdir + self.logname, "a") as logfile:
            logfile.write("Starting optimization at ")
            logfile.write(start_time_str)
            logfile.write("Optimization parameters:\n")
            logfile.write(hjson.dumpsJSON(self.pmap.opt_map, default=hjson_encode))
            logfile.write("\n")
            logfile.write("Units:\n")
            logfile.write(
                hjson.dumpsJSON(self.pmap.get_opt_units(), default=hjson_encode)
            )
            logfile.write("\n")
            logfile.write("Algorithm options:\n")
            logfile.write(hjson.dumpsJSON(self.options, default=hjson_encode))
            logfile.write("\n")
            logfile.flush()

        for logger in self.logger:
            logger.start_log(self, self.logdir)

    def end_log(self) -> None:
        """
        Finish the log by recording current time and total runtime.

        """
        self.end_time = time.time()
        with open(self.logdir + self.logname, "a") as logfile:
            logfile.write(f"Finished at {time.asctime(time.localtime())}\n")
            logfile.write(f"Total runtime: {self.end_time - self.start_time}\n\n")
            logfile.flush()

    def log_best_unitary(self) -> None:
        """
        Save the best unitary in the log.
        """
        with open(self.logdir + "best_point_" + self.logname, "w") as best_point:
            propagators = self.exp.propagators
            for gate, U in propagators.items():
                best_point.write("\n")
                best_point.write(f"Re {gate}: \n")
                best_point.write(f"{np.round(np.real(U), 3)}\n")
                best_point.write("\n")
                best_point.write(f"Im {gate}: \n")
                best_point.write(f"{np.round(np.imag(U), 3)}\n")

    def log_parameters(self) -> None:
        """
        Log the current status. Write parameters to log. Update the current best
        parameters. Call plotting functions as set up.

        """
        if self.optim_status["goal"] < self.current_best_goal:
            self.current_best_goal = self.optim_status["goal"]
            self.current_best_params = self.optim_status["params"]
            with open(self.logdir + "best_point_" + self.logname, "w") as best_point:
                best_dict = {
                    "opt_map": self.pmap.get_opt_map(),
                    "units": self.pmap.get_opt_units(),
                    "optim_status": self.optim_status,
                }
                best_point.write(hjson.dumps(best_dict, default=hjson_encode))
                best_point.write("\n")
        if self.store_unitaries:
            self.exp.store_Udict(self.optim_status["goal"])
            self.exp.store_unitaries_counter += 1

        with open(self.logdir + self.logname, "a") as logfile:
            logfile.write(
                f"\nFinished evaluation {self.evaluation} at {time.asctime()}\n"
            )
            # logfile.write(hjson.dumpsJSON(self.optim_status, indent=2))
            logfile.write(hjson.dumpsJSON(self.optim_status, default=hjson_encode))
            logfile.write("\n")
            logfile.flush()

        for logger in self.logger:
            logger.log_parameters(self.evaluation, self.optim_status)

    def goal_run(
        self, current_params: Union[np.ndarray, tf.constant]
    ) -> Union[np.ndarray, tf.constant]:
        """
        Placeholder for the goal function. To be implemented by inherited classes.
        """
        raise NotImplementedError("Implement this function in a subclass")

    def goal_run_with_grad(self, current_params):
        with tf.GradientTape(watch_accessed_variables=False) as t:
            t.watch(current_params)
            goal = self.goal_run(current_params)
        grad = t.gradient(goal, current_params)
        return goal, grad

    def lookup_gradient(self, x):
        """
        Return the stored gradient for a given parameter set.

        Parameters
        ----------
        x : np.array
            Parameter set.

        Returns
        -------
        np.array
            Value of the gradient.
        """
        key = str(x)
        gradient = self.gradients.pop(key)
        if np.any(np.isnan(gradient)) or np.any(np.isinf(gradient)):
            # TODO: is simply a warning sufficient?
            gradient[
                np.isnan(gradient)
            ] = 1e-10  # Most probably at boundary of Quantity
            gradient[
                np.isinf(gradient)
            ] = 1e-10  # Most probably at boundary of Quantity
        return gradient

    def fct_to_min(
        self, input_parameters: Union[np.ndarray, tf.constant]
    ) -> Union[np.ndarray, tf.constant]:
        """
        Wrapper for the goal function.

        Parameters
        ----------
        input_parameters : [np.array, tf.constant]
            Vector of parameters in the optimizer friendly way.

        Returns
        -------
        [np.ndarray, tf.constant]
            Value of the goal function. Float if input is np.array else tf.constant
        """

        if isinstance(input_parameters, np.ndarray):
            current_params = tf.constant(input_parameters)
            goal = self.goal_run(current_params)
            self.log_parameters()
            goal = float(goal)
            return goal
        else:
            current_params = input_parameters
            goal = self.goal_run(current_params)
            self.log_parameters()
            return goal

    def fct_to_min_autograd(self, x):
        """
         Wrapper for the goal function, including evaluation and storage of the
         gradient.

        Parameters
         ----------
         x : np.array
             Vector of parameters in the optimizer friendly way.

         Returns
         -------
         float
             Value of the goal function.
        """
        current_params = tf.constant(x)
        goal, grad = self.goal_run_with_grad(current_params)
        if isinstance(grad, tf.Tensor):
            grad = grad.numpy()
        gradients = grad.flatten()
        for i in tf.where(gradients == 0).numpy().tolist():
            warnings.warn(
                f"{self.pmap.get_key_from_scaled_index(i[0])} has no gradient. This might indicate no usage for current experiment.",
                Warning,
            )
        self.gradients[str(current_params.numpy())] = gradients
        self.optim_status["gradient"] = gradients.tolist()
        self.log_parameters()
        if isinstance(goal, tf.Tensor):
            goal = float(goal)
        return goal


class BaseLogger:
    def __init__(self):
        pass

    def start_log(self, opt, logdir):
        self.logdir = logdir

    def log_parameters(self, evaluation, optim_status):
        pass


class BestPointLogger(BaseLogger):
    pass


class TensorBoardLogger(BaseLogger):
    def __init__(self):
        super().__init__()
        self.opt_map = []
        self.writer = None
        self.store_better_iterations_only = True
        self.best_iteration = np.inf

    def write_params(self, params, step=0):
        assert len(self.opt_map) == len(params)
        for i in range(len(self.opt_map)):
            for key in self.opt_map[i]:
                if type(params[i]) is float:
                    tf.summary.scalar(key, float(params[i]), step=step)
                elif len(params[i]) == 1:
                    tf.summary.scalar(key, float(params[i][0]), step=step)
                else:
                    for jj in range(len(params[i])):
                        tf.summary.scalar(
                            key + "_" + str(jj), float(params[i][jj]), step=step
                        )

    def set_logdir(self, logdir):
        print("new Tensorboard Logdir")
        self.logdir = logdir

    def start_log(self, opt, logdir):
        self.opt_map = opt.pmap.get_opt_map()
        print("create log at", logdir)
        self.writer = tf.summary.create_file_writer(
            logdir=logdir,
        )

        with self.writer.as_default():
            self.write_params(opt.pmap.get_parameters())
            tf.summary.text(
                "Parameters",
                hjson.dumpsJSON(
                    opt.pmap.asdict(instructions_only=False),
                    indent=2,
                    default=hjson_encode,
                ),
                step=0,
            )

        self.writer.flush()
        hparams = dict()
        for k, hpar in opt.pmap.get_not_opt_params().items():
            val = hpar.numpy()
            if len(val.shape) < 1 or val.shape[0] < 2:
                hparams[k] = float(hpar)
            else:
                for i, v in enumerate(val.tolist()):
                    hparams[f"{k}_{i}"] = v
        with self.writer.as_default():
            hp.hparams(hparams)
        self.writer.flush()

    def log_parameters(self, evaluation, optim_status):
        if self.store_better_iterations_only:
            if optim_status["goal"] > self.best_iteration:
                return
            else:
                self.best_iteration = optim_status["goal"]
        opt_status = copy.deepcopy(optim_status)
        with self.writer.as_default():
            self.write_params(opt_status.pop("params"), evaluation)
            tf.summary.scalar("goal", opt_status.pop("goal"), step=evaluation)

            if "gradient" in opt_status:
                tf.summary.histogram(
                    "gradient",
                    tf.clip_by_value(opt_status.pop("gradient"), -3, 3),
                    step=evaluation,
                )

            opt_status.pop("time")
            for k, v in opt_status.items():
                if type(v) is float:
                    tf.summary.scalar(k, v, step=evaluation)
                elif type(v) is list:
                    tf.summary.histogram(k, v, step=evaluation)
                else:
                    # print(k, v)
                    raise Warning(
                        f"Elements of type {type(v)}, here {k}, are not yet implemented to be logged "
                    )
            self.writer.flush()
