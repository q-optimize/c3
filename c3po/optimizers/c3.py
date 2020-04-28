"""Object that deals with the model learning."""

import os
import json
import pickle
import itertools
import random
import numpy as np
import tensorflow as tf
from c3po.optimizers.optimizer import Optimizer
import matplotlib.pyplot as plt
from c3po.utils.utils import log_setup
import c3po.utils.display as display


class C3(Optimizer):
    """Object that deals with the model learning."""

    def __init__(
        self,
        dir_path,
        fom,
        sampling,
        batch_sizes,
        opt_map,
        state_labels=None,
        callback_foms=[],
        callback_figs=[],
        algorithm=None,
        run_name=None,
        options={}
    ):
        """Initiliase."""
        super().__init__(algorithm=algorithm)
        self.fom = fom
        self.sampling = sampling
        self.batch_sizes = batch_sizes
        self.opt_map = opt_map
        self.state_labels = state_labels
        self.callback_foms = callback_foms
        self.callback_figs = callback_figs
        self.inverse = False
        self.options = options
        self.learn_data = {}
        self.log_setup(dir_path, run_name)

    def log_setup(self, dir_path, run_name):
        self.dir_path = os.path.abspath(dir_path)
        if run_name is None:
            run_name = self.algorithm.__name__ + '-' \
                + self.sampling.__name__ + '-' \
                + self.fom.__name__
        # datafile = os.path.basename(self.datafile)
        # datafile = datafile.split('.')[0]
        # string = string + '----[' + datafile + ']'
        self.logdir = log_setup(self.dir_path, run_name)
        self.logname = 'model_learn.log'

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
        # TODO fix when batch size is 1 (atm it does all)
        learn_from = self.learn_from
        sampling = self.sampling
        indeces =  sampling(learn_from, batch_size)
        if self.inverse:
            return list(set(all) - set(indeces))
        else:
            return indeces

    def learn_model(self):
        self.start_log()
        self.nice_print = self.exp.print_parameters
        for cb_fig in self.callback_figs:
            os.makedirs(self.logdir + cb_fig.__name__)
        os.makedirs(self.logdir + 'dynamics_seq')
        os.makedirs(self.logdir + 'dynamics_xyxy')
        print(f"C3:STATUS:Saving as: {os.path.abspath(self.logdir + self.logname)}")
        x0 = self.exp.get_parameters(self.opt_map, scaled=True)
        try:
            # TODO deal with kears learning differently
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
        self.exp.set_parameters(best_params, self.opt_map)
        self.end_log()
        #self.confirm()

    def confirm(self):
        self.logname = 'confirm.log'
        self.inverse = True
        self.start_log()
        print(f"C3:STATUS:Saving as: {os.path.abspath(self.logdir + self.logname)}")
        x_best = self.exp.get_parameters(self.opt_map, scaled=True)
        self.evaluation = -1
        try:
            self.goal_run(x_best)
        except KeyboardInterrupt:
            pass

    def goal_run(self, current_params):
        # display.plot_C3([self.logdir])
        exp_values = []
        exp_stds = []
        sim_values = []
        exp_shots = []

        self.exp.set_parameters(current_params, self.opt_map, scaled=True)
        count = 0

        for target, data in self.learn_data.items():

            self.learn_from = data['seqs_grouped_by_param_set']
            self.gateset_opt_map = data['opt_map']
            indeces = self.select_from_data(self.batch_sizes[target])

            for ipar in indeces:

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
                    gateset_params, gateset_opt_map, scaled=False
                )
                # We find the unique gates used in the sequence and compute
                # only them.
                self.exp.opt_gates = list(
                    set(itertools.chain.from_iterable(sequences))
                )
                self.exp.get_gates()
                sim_vals = self.exp.evaluate(
                    sequences, labels=self.state_labels[target]
                )

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
                                f"{float(m_val[ii]-sim_val[ii]):-8.6f}\n"
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
        exp_shots = tf.constant(exp_shots, dtype=tf.float64)
        goal = self.fom(exp_values, sim_values, exp_stds, exp_shots)
        goal_numpy = float(goal.numpy())

        with open(self.logdir + self.logname, 'a') as logfile:
            logfile.write("\nFinished batch with ")
            logfile.write("{}: {}\n".format(self.fom.__name__, goal_numpy))
            print("{}: {}".format(self.fom.__name__, goal_numpy))
            for cb_fom in self.callback_foms:
                val = float(
                    cb_fom(exp_values, sim_values, exp_stds, exp_shots).numpy()
                )
                logfile.write("{}: {}\n".format(cb_fom.__name__, val))
                print("{}: {}".format(cb_fom.__name__, val))
            print("")
            logfile.flush()

        for cb_fig in self.callback_figs:
            fig = cb_fig(exp_values, sim_values.numpy(), exp_stds)
            fig.savefig(
                self.logdir
                + cb_fig.__name__ + '/'
                + 'eval:' + str(self.evaluation) + "__"
                + self.fom.__name__ + str(round(goal_numpy, 3))
                + '.png'
            )
            plt.close(fig)

        self.optim_status['params'] = [
            par.numpy().tolist()
            for par in self.exp.get_parameters(self.opt_map)
        ]
        self.optim_status['goal'] = goal_numpy
        self.evaluation += 1
        return goal
