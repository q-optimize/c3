"""Object that deals with the model learning."""

import os
import time
import json
import pickle
import itertools
import random
import numpy as np
import tensorflow as tf
from c3.optimizers.optimizer import Optimizer
import matplotlib.pyplot as plt
from c3.utils.utils import log_setup
import c3.utils.display as display
from c3.libraries.estimators import dv_g_LL_prime, g_LL_prime_combined, g_LL_prime, neg_loglkh_multinom_norm


class C3(Optimizer):
    """
    Object that deals with the model learning.

    Parameters
    ----------
    dir_path : str
        Filepath to save results
    fom : callable
        Figure of merit from the estimator library
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
    callback_figs : list
        List of plotting functions to run at every evaluation
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
        fom,
        sampling,
        batch_sizes,
        seqs_per_point,
        pmap,
        state_labels=None,
        callback_foms=[],
        callback_figs=[],
        algorithm=None,
        run_name=None,
        options={}
    ):
        """Initiliase.
        """
        super().__init__(algorithm=algorithm)
        self.fom = fom
        self.sampling = sampling
        self.batch_sizes = batch_sizes
        self.seqs_per_point = seqs_per_point
        self.pmap = pmap
        self.state_labels = state_labels
        self.callback_foms = callback_foms
        self.callback_figs = callback_figs
        self.inverse = False
        self.options = options
        self.learn_data = {}
        self.log_setup(dir_path, run_name)

    def log_setup(self, dir_path, run_name):
        """
        Create the folders to store data.

        Parameters
        ----------
        dir_path : str
            Filepath
        run_name : str
            User specified name for the run

        """
        self.dir_path = os.path.abspath(dir_path)
        if run_name is None:
            run_name = self.algorithm.__name__ + '-' \
                + self.sampling.__name__ + '-' \
                + self.fom.__name__
        self.logdir = log_setup(self.dir_path, run_name)
        self.logname = 'model_learn.log'

    def read_data(self, datafiles):
        """
        Open data files and read in experiment results.

        Parameters
        ----------
        datafiles : list of str
            List of paths for files that contain learning data.
        """
        for target, datafile in datafiles.items():
            with open(datafile, 'rb+') as file:
                self.learn_data[target] = pickle.load(file)

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
            best_exp_opt_map = [tuple(a) for a in json.loads(best[0])]
            init_p = json.loads(best[1])['params']
            self.pmap.set_parameters(init_p, best_exp_opt_map)
            self.pmap.model.update_model()

    def select_from_data(self, batch_size):
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
        all = list(range(total_size))
        if self.inverse:
            return list(set(all) - set(indeces))
        else:
            return indeces

    def learn_model(self):
        """
        Peroms the model learning by minimizing the figure of merit.
        """
        self.start_log()
        for cb_fig in self.callback_figs:
            os.makedirs(self.logdir + cb_fig.__name__)
        # os.makedirs(self.logdir + 'dynamics_seq')
        # os.makedirs(self.logdir + 'dynamics_xyxy')
        print(f"C3:STATUS:Saving as: {os.path.abspath(self.logdir + self.logname)}")
        x0 = self.pmap.get_parameters_scaled()
        # TODO Nico: Store initial parameters to recover them later, do we need this?
        self.init_params = self.pmap.gateset.get_parameters()
        self.init_opt_map = self.pmap.gateset.list_parameters()
        try:
            # TODO deal with keras learning differently
            self.algorithm(
                x0,
                fun=self.fct_to_min,
                fun_grad=self.fct_to_min_autograd,
                grad_lookup=self.lookup_gradient,
                options=self.options
            )
        except KeyboardInterrupt:
            pass
        # display.plot_C3([self.logdir])
        with open(self.logdir + 'best_point_' + self.logname, 'r') as file:
            best_params = json.loads(file.readlines()[1])['params']
        self.pmap.set_parameters(best_params)
        self.pmap.model.update_model()
        self.end_log()
        self.confirm()

    def confirm(self):
        """
        Compute the validation set, i.e. the value of the goal function on all points of the dataset that were not used
        for learning.
                """
        self.logname = 'confirm.log'
        self.inverse = True
        self.start_log()
        print(f"C3:STATUS:Saving as: {os.path.abspath(self.logdir + self.logname)}")
        x_best = self.pmap.get_parameters_scaled()
        self.evaluation = -1
        try:
            self.goal_run(x_best)
        except KeyboardInterrupt:
            pass

    def goal_run(self, current_params):
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
        # display.plot_C3([self.logdir], only_iterations=False)
        exp_values = []
        sim_values = []
        exp_stds = []
        exp_shots = []
        goals = []
        grads = []
        seq_weigths = []
        count = 0
        seqs_pp = self.seqs_per_point
        #TODO: seq per point is not constant. Remove.

        for target, data in self.learn_data.items():

            self.learn_from = data['seqs_grouped_by_param_set']
            self.gateset_opt_map = data['opt_map']
            indeces = self.select_from_data(self.batch_sizes[target])

            for ipar in indeces:

                count += 1
                m = self.learn_from[ipar]
                gateset_params = m['params']
                gateset_opt_map = self.gateset_opt_map
                m_vals = m['results'][:seqs_pp]
                m_stds = m['results_std'][:seqs_pp]
                m_shots = m['shots'][:seqs_pp]
                sequences = m['seqs'][:seqs_pp]
                num_seqs = len(sequences)
                if target == 'all':
                    num_seqs = len(sequences) * 3

                self.pmap.set_parameters_scaled(current_params)
                self.pmap.model.update_model()

                # We make sure to reset the control parameters
                self.exp.gateset.set_parameters(
                    self.init_gateset_params,
                    self.init_gateset_opt_map
                )
                self.exp.gateset.set_parameters(
                    gateset_params, gateset_opt_map, scaled=False
                )
                # We find the unique gates used in the sequence and compute
                # only them.
                self.exp.opt_gates = list(
                    set(itertools.chain.from_iterable(sequences))
                )
                self.exp.get_gates()
                self.exp.evaluate(sequences)
                sim_vals = self.exp.process(labels=self.state_labels[target])

                exp_stds.extend(m_stds)
                exp_shots.extend(m_shots)

                if target == 'all':
                    goal = neg_loglkh_multinom_norm(
                        m_vals,
                        tf.stack(sim_vals),
                        tf.constant(m_stds, dtype=tf.float64),
                        tf.constant(m_shots, dtype=tf.float64)
                    )
                else:
                    goal = g_LL_prime(
                        m_vals,
                        tf.stack(sim_vals),
                        tf.constant(m_stds, dtype=tf.float64),
                        tf.constant(m_shots, dtype=tf.float64)
                    )
                goals.append(goal.numpy())
                seq_weigths.append(num_seqs)
                sim_values.extend(sim_vals)
                exp_values.extend(m_vals)

                with open(self.logdir + self.logname, 'a') as logfile:
                    logfile.write(
                        "\n  Parameterset {}, #{} of {}:\n {}\n {}\n".format(
                            ipar + 1,
                            count,
                            len(indeces),
                            json.dumps(self.gateset_opt_map),
                            self.exp.gateset.get_parameters(
                                self.gateset_opt_map, to_str=True
                            ),
                        )
                    )
                    logfile.write(
                        "Sequence    Simulation  Experiment  Std           Shots"
                        "    Diff\n"
                    )

                for iseq in range(len(sequences)):
                    m_val = np.array(m_vals[iseq])
                    m_std = np.array(m_stds[iseq])
                    shots = np.array(m_shots[iseq])
                    sim_val = sim_vals[iseq].numpy()
                    int_len = len(str(num_seqs))
                    with open(self.logdir + self.logname, 'a') as logfile:
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

        goal = g_LL_prime_combined(goals, seq_weigths)
        # TODO make gradient free function use any fom

        with open(self.logdir + self.logname, 'a') as logfile:
            logfile.write("\nFinished batch with ")
            logfile.write("{}: {}\n".format(self.fom.__name__, goal))
            # print("{}: {}".format(self.fom.__name__, goal))
            for cb_fom in self.callback_foms:
                val = float(
                    cb_fom(exp_values, sim_values, exp_stds, exp_shots).numpy()
                )
                logfile.write("{}: {}\n".format(cb_fom.__name__, val))
                # print("{}: {}".format(cb_fom.__name__, val))
            # print("")
            logfile.flush()

        for cb_fig in self.callback_figs:
            fig = cb_fig(exp_values, sim_values.numpy(), exp_stds)
            fig.savefig(
                self.logdir
                + cb_fig.__name__ + '/'
                + 'eval:' + str(self.evaluation) + "__"
                + self.fom.__name__ + str(round(goal, 3))
                + '.png'
            )
            plt.close(fig)

        self.optim_status['params'] = [
            par.numpy().tolist()
            for par in self.exp.get_parameters(self.opt_map)
        ]
        self.optim_status['goal'] = goal
        self.optim_status['time'] = time.asctime()
        self.evaluation += 1
        return goal

    def goal_run_with_grad(self, current_params):
        """
        Same as goal_run but with gradient. Very resource intensive. Unoptimized at the moment.
        """
        # display.plot_C3([self.logdir])
        exp_values = []
        sim_values = []
        exp_stds = []
        exp_shots = []
        goals = []
        grads = []
        seq_weigths = []
        count = 0
        seqs_pp = self.seqs_per_point

        for target, data in self.learn_data.items():

            self.learn_from = data['seqs_grouped_by_param_set']
            self.gateset_opt_map = data['opt_map']
            indeces = self.select_from_data(self.batch_sizes[target])

            for ipar in indeces:

                count += 1
                m = self.learn_from[ipar]
                gateset_params = m['params']
                gateset_opt_map = self.gateset_opt_map
                m_vals = m['results'][:seqs_pp]
                m_stds = np.array(m['results_std'][:seqs_pp])
                m_shots = m['shots'][:seqs_pp]
                sequences = m['seqs'][:seqs_pp]
                num_seqs = len(sequences)
                if target == 'all':
                    num_seqs = len(sequences) * 3

                with tf.GradientTape() as t:
                    t.watch(current_params)
                    self.exp.set_parameters(current_params, self.opt_map, scaled=True)
                    # TODO Update model
                    self.exp.gateset.set_parameters(
                        self.init_gateset_params,
                        self.init_gateset_opt_map,
                        scaled=False
                    )
                    self.exp.gateset.set_parameters(
                        gateset_params, gateset_opt_map, scaled=False
                    )
                    # We find the unique gates used in the sequence and compute
                    # only them.
                    self.exp.opt_gates = list(
                        set(itertools.chain.from_iterable(sequences))
                    )
                    self.exp.get_gates()
                    self.exp.evaluate(sequences)
                    sim_vals = self.exp.process(labels=self.state_labels[target])

                    exp_stds.extend(m_stds)
                    exp_shots.extend(m_shots)

                    if target == 'all':
                        g = neg_loglkh_multinom_norm(
                            m_vals,
                            tf.stack(sim_vals),
                            tf.constant(m_stds, dtype=tf.float64),
                            tf.constant(m_shots, dtype=tf.float64)
                        )
                    else:
                        g = g_LL_prime(
                            m_vals,
                            tf.stack(sim_vals),
                            tf.constant(m_stds, dtype=tf.float64),
                            tf.constant(m_shots, dtype=tf.float64)
                        )

                seq_weigths.append(num_seqs)
                goals.append(g.numpy())
                grads.append(t.gradient(g, current_params).numpy())

                sim_values.extend(sim_vals)
                exp_values.extend(m_vals)

                with open(self.logdir + self.logname, 'a') as logfile:
                    logfile.write(
                        "\n  Parameterset {}, #{} of {}:\n {}\n {}\n".format(
                            ipar + 1,
                            count,
                            len(indeces),
                            json.dumps(self.gateset_opt_map),
                            self.exp.gateset.get_parameters(
                                self.gateset_opt_map, to_str=True
                            ),
                        )
                    )
                    logfile.write(
                        "Sequence    Simulation  Experiment  Std         Shots"
                        "       Diff\n"
                    )

                for iseq in range(len(sequences)):
                    m_val = np.array(m_vals[iseq])
                    m_std = np.array(m_stds[iseq])
                    shots = np.array(m_shots[iseq])
                    sim_val = sim_vals[iseq].numpy()
                    int_len = len(str(num_seqs))
                    with open(self.logdir + self.logname, 'a') as logfile:
                        for ii in range(len(sim_val)):
                            logfile.write(
                                f"{iseq + 1:8}    "
                                f"{float(sim_val[ii]):8.6f}    "
                                f"{float(m_val[ii]):8.6f}    "
                                f"{float(m_std[ii]):8.6f}    "
                                f"{float(shots[0]):8}    "
                                f"{float(m_val[ii]-sim_val[ii]):8.6f}\n"
                            )
                        logfile.flush()


        # exp_values = tf.constant(exp_values, dtype=tf.float64)
        # sim_values =  tf.stack(sim_values)
        # exp_stds = tf.constant(exp_stds, dtype=tf.float64)
        # exp_shots = tf.constant(exp_shots, dtype=tf.float64)

        goal = g_LL_prime_combined(goals, seq_weigths)
        grad = dv_g_LL_prime(goals, grads, seq_weigths)

        with open(self.logdir + self.logname, 'a') as logfile:
            logfile.write("\nFinished batch with ")
            logfile.write("{}: {}\n".format(self.fom.__name__, goal))
            # print("{}: {}".format(self.fom.__name__, goal))
            for cb_fom in self.callback_foms:
                val = float(
                    cb_fom(exp_values, sim_values, exp_stds, exp_shots).numpy()
                )
                logfile.write("{}: {}\n".format(cb_fom.__name__, val))
                # print("{}: {}".format(cb_fom.__name__, val))
            # print("")
            logfile.flush()

        for cb_fig in self.callback_figs:
            fig = cb_fig(exp_values, sim_values.numpy(), exp_stds)
            fig.savefig(
                self.logdir
                + cb_fig.__name__ + '/'
                + 'eval:' + str(self.evaluation) + "__"
                + self.fom.__name__ + str(round(goal, 3))
                + '.png'
            )
            plt.close(fig)

        self.optim_status['params'] = [
            par.numpy().tolist()
            for par in self.exp.get_parameters(self.opt_map)
        ]
        self.optim_status['goal'] = goal
        self.optim_status['gradient'] = list(grad.flatten())
        self.optim_status['time'] = time.asctime()
        self.evaluation += 1
        return goal, grad
