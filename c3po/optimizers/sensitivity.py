"""Object that deals with the sensitivity test."""

import os
import json
import pickle
import itertools
import time
import numpy as np
import tensorflow as tf
import c3po.utils.display as display
from c3po.optimizers.optimizer import Optimizer
import matplotlib.pyplot as plt
from c3po.utils.utils import log_setup


class SET(Optimizer):
    """Object that deals with the sensitivity test."""

    def __init__(
        self,
        dir_path,
        fom,
        estimator_list,
        sampling,
        batch_sizes,
        state_labels=None,
        sweep_map=None,
        sweep_bounds=None,
        algorithm=None,
        run_name=None,
        same_dyn=False,
        options={}
    ):
        """Initiliase."""
        super().__init__(algorithm=algorithm)
        self.fom = fom
        self.estimator_list = estimator_list
        self.sampling = sampling
        self.batch_sizes = batch_sizes
        self.state_labels = state_labels
        self.opt_map = sweep_map
        self.sweep_bounds = sweep_bounds
        self.options = options
        self.inverse = False
        self.learn_data = {}
        self.same_dyn = same_dyn
        self.log_setup(dir_path, run_name)

    def log_setup(self, dir_path, run_name):
        self.dir_path = os.path.abspath(dir_path)
        if run_name is None:
            run_name = "sensitivity" \
                + self.algorithm.__name__ + '-' \
                + self.sampling.__name__ + '-' \
                + self.fom.__name__
        self.logdir = log_setup(self.dir_path, run_name)
        self.logname = "sensitivity.log"

    def read_data(self, datafiles):
        for target, datafile in datafiles.items():
            with open(datafile, 'rb+') as file:
                self.learn_data[target] = pickle.load(file)

    def load_best(self, init_point):
        with open(init_point) as init_file:
            best = init_file.readlines()
            best_exp_opt_map = [tuple(a) for a in json.loads(best[0])]
            init_p = json.loads(best[1])['params']
            self.exp.set_parameters(init_p, best_exp_opt_map)

    def select_from_data(self, batch_size):
        learn_from = self.learn_from
        sampling = self.sampling
        indeces =  sampling(learn_from, batch_size)
        if self.inverse:
            return list(set(all) - set(indeces))
        else:
            return indeces

    def sensitivity_test(self):
        self.start_log()
        self.nice_print = self.exp.print_parameters

        print("Initial parameters:")
        print(self.exp.print_parameters())
        self.dfname = "data.dat"
        self.options['bounds'] = self.sweep_bounds

        print(f"C3:STATUS:Saving as: {os.path.abspath(self.logdir + self.logname)}")
        x0 = self.exp.get_parameters(self.opt_map, scaled=True)
        self.init_gateset_params = self.exp.gateset.get_parameters()
        self.init_gateset_opt_map = self.exp.gateset.list_parameters()
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

        # #=== Get the resulting data ======================================

        # Xs=np.array(list(learner.data.keys()))
        # Ys=np.array(list(learner.data.values()))
        # Ks=np.argsort(Xs)
        # Xs=Xs[Ks]
        # Ys=Ys[Ks]

    def goal_run(self, val):
        exp_values = []
        exp_stds = []
        sim_values = []
        exp_shots = []

        # print("tup: " + str(tup))
        # print("val: " + str(val))
        # print(self.opt_map)
        self.exp.set_parameters(val, self.opt_map, scaled=False)
        # print("params>>> ")
        # print(self.exp.print_parameters(self.opt_map))

        # print("self.learn_data.items(): " + str(len(self.learn_data.items())))
        count = 0
        for target, data in self.learn_data.items():

            self.learn_from = data['seqs_grouped_by_param_set']
            self.gateset_opt_map = data['opt_map']
            indeces = self.select_from_data(self.batch_sizes[target])

            for ipar in indeces:
                # if count % 100 == 0:
                #     print("count: " + str(count))

                count += 1
                m = self.learn_from[ipar]
                gateset_params = m['params']
                gateset_opt_map = self.gateset_opt_map
                m_vals = m['results']
                m_stds = m['results_std']
                m_shots = m['shots']
                sequences = m['seqs']
                num_seqs = len(sequences)

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
                if self.same_dyn and self.evaluation != 0:
                    pass
                else:
                    self.exp.get_gates()
                    self.exp.evaluate(sequences)
                sim_vals = self.exp.process(labels=self.state_labels[target])

                # exp_values.extend(m_vals)
                # exp_stds.extend(m_stds)
                sim_values.extend(sim_vals)

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

                for iseq in range(num_seqs):
                    m_val = np.array(m_vals[iseq])
                    m_std = np.array(m_stds[iseq])
                    shots = np.array(m_shots[iseq])
                    exp_values.append(m_val)
                    exp_stds.append(m_std)
                    exp_shots.append(shots)
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

        exp_values = tf.constant(exp_values, dtype=tf.float64)
        sim_values =  tf.stack(sim_values)
        if exp_values.shape != sim_values.shape:
            print(
                "C3:WARNING:"
                "Data format of experiment and simulation figures of"
                " merit does not match."
            )
        exp_stds = tf.constant(exp_stds, dtype=tf.float64)
#        print("exp_shots: " + str(exp_shots))
        exp_shots = tf.constant(exp_shots, dtype=tf.float64)
        goal = self.fom(exp_values, sim_values, exp_stds, exp_shots)
        goal_numpy = float(goal.numpy())

        with open(self.logdir + self.dfname, 'a') as datafile:
            datafile.write(f"{val}\t{goal_numpy}\n")

        for est in self.estimator_list:
            tmp = est(exp_values, sim_values, exp_stds, exp_shots)
            tmp = float(tmp.numpy())
            fname = est.__name__ + '.dat'
            with open(self.logdir + fname, 'a') as datafile:
                datafile.write(f"{val}\t{tmp}\n")

        with open(self.logdir + self.logname, 'a') as logfile:
            logfile.write("\nFinished batch with ")
            logfile.write("{}: {}\n".format(self.fom.__name__, goal_numpy))
            print("{}: {}".format(self.fom.__name__, goal_numpy))
            for est in self.estimator_list:
                val = float(
                    est(exp_values, sim_values, exp_stds, exp_shots).numpy()
                )
                logfile.write("{}: {}\n".format(est.__name__, val))
                #print("{}: {}".format(est.__name__, val))
            print("")
            logfile.flush()

        self.optim_status['params'] = [
            par.numpy().tolist()
            for par in self.exp.get_parameters(self.opt_map)
        ]
        self.optim_status['goal'] = goal_numpy
        self.evaluation += 1
        return goal
