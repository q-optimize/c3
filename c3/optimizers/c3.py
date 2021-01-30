"""Object that deals with the model learning."""

import os
import time
import hjson
import pickle
import itertools
import numpy as np
import tensorflow as tf
from typing import List, Dict
from c3.optimizers.optimizer import Optimizer
from c3.utils.utils import log_setup
from c3.libraries.estimators import (
    dv_g_LL_prime,
    g_LL_prime_combined,
    g_LL_prime,
    neg_loglkh_multinom_norm,
)


class C3(Optimizer):
    """
    Object that deals with the model learning.

    Parameters
    ----------
    dir_path : str
        Filepath to save results
    sampling : str
        Sampling method from the sampling library
    batch_sizes : list
        Number of points to select from each dataset
    seqs_per_point : int
        Number of sequences that use the same parameter set
    pmap : ParameterMap
        Identifiers for the parameter vector
    state_labels : list
        Identifiers for the qubit subspaces
    callback_foms : list
        Figures of merit to additionally compute and store
    algorithm : callable
        From the algorithm library
    run_name : str
        User specified name for the run, will be used as root folder
    options : dict
        Options to be passed to the algorithm
    """

    def __init__(
        self,
        dir_path,
        sampling,
        batch_sizes,
        pmap,
        seqs_per_point=None,
        state_labels=None,
        callback_foms=[],
        algorithm=None,
        run_name=None,
        options={},
    ):
        """Initiliase."""
        super().__init__(pmap=pmap, algorithm=algorithm)
        self.sampling = sampling
        self.batch_sizes = batch_sizes
        self.seqs_per_point = seqs_per_point
        self.state_labels = state_labels
        self.callback_foms = callback_foms
        self.inverse = False
        self.options = options
        self.learn_data = {}
        self.fom = g_LL_prime_combined
        self.__dir_path = dir_path
        self.__run_name = run_name

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
            run_name = "-".join(
                [self.algorithm.__name__, self.sampling.__name__, self.fom.__name__]
            )
        self.logdir = log_setup(dir_path, run_name)
        self.logname = "model_learn.log"
        # shutil.copy2(self.__real_model_folder, self.logdir)

    def read_data(self, datafiles: Dict[str, str]) -> None:
        """
        Open data files and read in experiment results.

        Parameters
        ----------
        datafiles : dict
            List of paths for files that contain learning data.
        """
        self.__real_model_folder = os.path.dirname(list(datafiles.values())[0])
        for target, datafile in datafiles.items():
            with open(datafile, "rb+") as file:
                self.learn_data[target] = pickle.load(file)

    def select_from_data(self, batch_size) -> List[int]:
        """
        Select a subset of each dataset to compute the goal function on.

        Parameters
        ----------
        batch_size : int
            Number of points to select

        Returns
        -------
        list
            Indeces of the selected data points.
        """
        # TODO fix when batch size is 1 (atm it does all)
        learn_from = self.learn_from
        sampling = self.sampling
        indeces = sampling(learn_from, batch_size)
        total_size = len(learn_from)
        all_indxs = list(range(total_size))
        if self.inverse:
            selected_indeces = list(set(all_indxs) - set(indeces))
        else:
            selected_indeces = indeces
        return selected_indeces

    def learn_model(self) -> None:
        """
        Performs the model learning by minimizing the figure of merit.
        """
        self.log_setup()
        self.start_log()
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
        self.pmap.model.update_model()
        self.end_log()
        self.confirm()

    def confirm(self) -> None:
        """
        Compute the validation set, i.e. the value of the goal function on all points
        of the dataset that were not used for learning.
        """
        self.logname = "confirm.log"
        self.inverse = True
        self.start_log()
        print(f"C3:STATUS:Saving as: {os.path.abspath(self.logdir + self.logname)}")
        x_best = self.pmap.get_parameters_scaled()
        self.evaluation = -1
        try:
            self.goal_run(x_best)
        except KeyboardInterrupt:
            pass

    def _one_par_sim_vals(
        self, current_params: tf.Variable, m: dict, ipar: int, target: str
    ) -> tf.float64:
        seqs_pp = self.seqs_per_point
        m = self.learn_from[ipar]
        gateset_params = m["params"]
        m_vals = m["results"][:seqs_pp]
        m_stds = np.array(m["result_stds"][:seqs_pp])
        m_shots = m["shots"][:seqs_pp]
        sequences = m["seqs"][:seqs_pp]
        if target == "all":
            num_seqs = len(sequences) * 3
        self.pmap.set_parameters_scaled(current_params)
        self.pmap.str_parameters()
        self.pmap.model.update_model()
        self.pmap.set_parameters(gateset_params, self.gateset_opt_map)
        # We find the unique gates used in the sequence and compute
        # only those.
        self.exp.opt_gates = list(set(itertools.chain.from_iterable(sequences)))
        self.exp.get_gates()
        pops = self.exp.evaluate(sequences)
        sim_vals, pops = self.exp.process(
            labels=self.state_labels[target], populations=pops
        )
        return sim_vals

    def _log_one_dataset(
        self, data_set: dict, ipar: int, indeces: list, sim_vals: list, count: int
    ) -> None:
        seqs_pp = self.seqs_per_point
        m_vals = data_set["results"][:seqs_pp]
        m_stds = np.array(data_set["result_stds"][:seqs_pp])
        m_shots = data_set["shots"][:seqs_pp]
        sequences = data_set["seqs"][:seqs_pp]
        with open(self.logdir + self.logname, "a") as logfile:
            logfile.write(
                f"\n  Parameterset {ipar + 1}, #{count} of {len(indeces)}:\n"
                f"{str(self.exp.pmap)}\n"
            )
            logfile.write(
                "Sequence    Simulation  Experiment  Std           Shots" "    Diff\n"
            )

        for iseq in range(len(sequences)):
            m_val = np.array(m_vals[iseq])
            m_std = np.array(m_stds[iseq])
            shots = np.array(m_shots[iseq])
            sim_val = sim_vals[iseq].numpy()
            with open(self.logdir + self.logname, "a") as logfile:
                for ii in range(len(sim_val)):
                    logfile.write(
                        f"{iseq + 1:8}    "
                        f"{float(sim_val[ii]):8.6f}    "
                        f"{float(m_val[ii]):8.6f}    "
                        f"{float(m_std[ii]):8.6f}    "
                        f"{float(shots[0]):8}     "
                        f"{float(m_val[ii]-sim_val[ii]): 8.6f}\n"
                    )
                logfile.flush()

    def goal_run(self, current_params: tf.Variable) -> tf.float64:
        """
        Evaluate the figure of merit for the current model parameters.

        Parameters
        ----------
        current_params : tf.Tensor
            Current model parameters

        Returns
        -------
        tf.float64
            Figure of merit

        """
        exp_values = []
        sim_values = []
        exp_stds = []
        exp_shots = []
        goals = []
        seq_weigths = []
        count = 0
        seqs_pp = self.seqs_per_point
        # TODO: seq per point is not constant. Remove.

        for target, data in self.learn_data.items():

            self.learn_from = data["seqs_grouped_by_param_set"]
            self.gateset_opt_map = data["opt_map"]
            indeces = self.select_from_data(self.batch_sizes[target])

            for ipar in indeces:

                count += 1
                data_set = self.learn_from[ipar]
                m_vals = data_set["results"][:seqs_pp]
                m_stds = data_set["result_stds"][:seqs_pp]
                m_shots = data_set["shots"][:seqs_pp]
                sequences = data_set["seqs"][:seqs_pp]
                num_seqs = len(sequences)
                if target == "all":
                    num_seqs = len(sequences) * 3

                sim_vals = self._one_par_sim_vals(
                    current_params, data_set, ipar, target
                )
                if target == "all":
                    one_goal = neg_loglkh_multinom_norm(
                        m_vals,
                        tf.stack(sim_vals),
                        tf.Variable(m_stds, dtype=tf.float64),
                        tf.Variable(m_shots, dtype=tf.float64),
                    )
                else:
                    one_goal = g_LL_prime(
                        m_vals,
                        tf.stack(sim_vals),
                        tf.Variable(m_stds, dtype=tf.float64),
                        tf.Variable(m_shots, dtype=tf.float64),
                    )
                exp_stds.extend(m_stds)
                exp_shots.extend(m_shots)

                goals.append(one_goal.numpy())
                seq_weigths.append(num_seqs)
                sim_values.extend(sim_vals)
                exp_values.extend(m_vals)

                self._log_one_dataset(data_set, ipar, indeces, sim_vals, count)

        goal = g_LL_prime_combined(goals, seq_weigths)
        # TODO make gradient free function use any fom

        with open(self.logdir + self.logname, "a") as logfile:
            logfile.write("\nFinished batch with ")
            logfile.write("{}: {}\n".format("g_LL_prime_combined", goal))
            for cb_fom in self.callback_foms:
                val = float(cb_fom(exp_values, sim_values, exp_stds, exp_shots).numpy())
                logfile.write("{}: {}\n".format(cb_fom.__name__, val))
            logfile.flush()

        self.optim_status["params"] = [
            par.numpy().tolist() for par in self.pmap.get_parameters()
        ]
        self.optim_status["goal"] = goal
        self.optim_status["time"] = time.asctime()
        self.evaluation += 1
        return goal

    def goal_run_with_grad(self, current_params):
        """
        Same as goal_run but with gradient. Very resource intensive. Unoptimized at the
        moment.
        """
        exp_values = []
        sim_values = []
        exp_stds = []
        exp_shots = []
        goals = []
        grads = []
        seq_weigths = []
        count = 0

        for target, data in self.learn_data.items():

            self.learn_from = data["seqs_grouped_by_param_set"]
            self.gateset_opt_map = data["opt_map"]
            indeces = self.select_from_data(self.batch_sizes[target])
            for ipar in indeces:

                count += 1
                data_set = self.learn_from[ipar]
                with tf.GradientTape() as t:
                    t.watch(current_params)
                    sim_vals = self._one_par_sim_vals(
                        current_params, data_set, ipar, target
                    )

                    if target == "all":
                        one_goal = neg_loglkh_multinom_norm(
                            m_vals,
                            tf.stack(sim_vals),
                            tf.Variable(m_stds, dtype=tf.float64),
                            tf.Variable(m_shots, dtype=tf.float64),
                        )
                    else:
                        one_goal = g_LL_prime(
                            m_vals,
                            tf.stack(sim_vals),
                            tf.Variable(m_stds, dtype=tf.float64),
                            tf.Variable(m_shots, dtype=tf.float64),
                        )

                seqs_pp = self.seqs_per_point
                m_vals = data_set["results"][:seqs_pp]
                m_stds = np.array(data_set["result_stds"][:seqs_pp])
                m_shots = data_set["shots"][:seqs_pp]
                sequences = data_set["seqs"][:seqs_pp]
                num_seqs = len(sequences)
                if target == "all":
                    num_seqs = len(sequences) * 3
                exp_stds.extend(m_stds)
                exp_shots.extend(m_shots)
                seq_weigths.append(num_seqs)

                goals.append(one_goal.numpy())
                grads.append(t.gradient(one_goal, current_params).numpy())

                sim_values.extend(sim_vals)
                exp_values.extend(m_vals)
                self._log_one_dataset(data_set, ipar, indeces, sim_vals, count)

        goal = g_LL_prime_combined(goals, seq_weigths)
        grad = dv_g_LL_prime(goals, grads, seq_weigths)
        # print(f"{seq_weigths=}\n{goals=}\n{grads=}\n{goal=}\n{grad=}\n")

        with open(self.logdir + self.logname, "a") as logfile:
            logfile.write("\nFinished batch with ")
            logfile.write("{}: {}\n".format(self.fom.__name__, goal))
            for cb_fom in self.callback_foms:
                val = float(cb_fom(exp_values, sim_values, exp_stds, exp_shots).numpy())
                logfile.write("{}: {}\n".format(cb_fom.__name__, val))
            logfile.flush()

        self.optim_status["params"] = [
            par.numpy().tolist() for par in self.pmap.get_parameters()
        ]
        self.optim_status["goal"] = goal
        self.optim_status["gradient"] = list(grad.flatten())
        self.optim_status["time"] = time.asctime()
        self.evaluation += 1
        return goal, grad

    def goal_run_with_grad_no_batch(self, current_params):
        """
        Same as goal_run but with gradient. Very resource intensive. Unoptimized at the
        moment.
        """
        exp_values = []
        sim_values = []
        exp_stds = []
        exp_shots = []
        count = 0
        seqs_pp = self.seqs_per_point

        with tf.GradientTape() as t:
            t.watch(current_params)
            for target, data in self.learn_data.items():
                self.learn_from = data["seqs_grouped_by_param_set"]
                self.gateset_opt_map = data["opt_map"]
                indeces = self.select_from_data(self.batch_sizes[target])
                for ipar in indeces:
                    count += 1
                    data_set = self.learn_from[ipar]
                    m_vals = data_set["results"][:seqs_pp]
                    sim_vals = self._one_par_sim_vals(
                        current_params, data_set, ipar, target
                    )
                    sim_values.extend(sim_vals)
                    exp_values.extend(m_vals)

                    self._log_one_dataset(data_set, ipar, indeces, sim_vals, count)

            if target == "all":
                goal = neg_loglkh_multinom_norm(
                    exp_values,
                    tf.stack(sim_values),
                    tf.Variable(exp_stds, dtype=tf.float64),
                    tf.Variable(exp_shots, dtype=tf.float64),
                )
            else:
                goal = g_LL_prime(
                    exp_values,
                    tf.stack(sim_values),
                    tf.Variable(exp_stds, dtype=tf.float64),
                    tf.Variable(exp_shots, dtype=tf.float64),
                )
            grad = t.gradient(goal, current_params).numpy()
            goal = goal.numpy()

        with open(self.logdir + self.logname, "a") as logfile:
            logfile.write("\nFinished batch with ")
            logfile.write("{}: {}\n".format(self.fom.__name__, goal))
            for cb_fom in self.callback_foms:
                val = float(cb_fom(exp_values, sim_values, exp_stds, exp_shots).numpy())
                logfile.write("{}: {}\n".format(cb_fom.__name__, val))
            logfile.flush()

        self.optim_status["params"] = [
            par.numpy().tolist() for par in self.pmap.get_parameters()
        ]
        self.optim_status["goal"] = goal
        self.optim_status["gradient"] = list(grad.flatten())
        self.optim_status["time"] = time.asctime()
        self.evaluation += 1
        return goal, grad
