"""Object that deals with the model learning."""

import os
import json
import pickle
import numpy as np
import tensorflow as tf
import c3po.utils.display as display
from c3po.optimizers.optimizer import Optimizer
import matplotlib.pyplot as plt
from c3po.utils.utils import log_setup


class C3(Optimizer):
    """Object that deals with the model learning."""

    def __init__(
        self,
        dir_path,
        fom,
        sampling,
        batch_size,
        opt_map,
        callback_foms=[],
        callback_figs=[],
        algorithm_no_grad=None,
        algorithm_with_grad=None,
        options={}
    ):
        """Initiliase."""
        super().__init__(
            algorithm_no_grad=algorithm_no_grad,
            algorithm_with_grad=algorithm_with_grad
            )
        self.fom = fom
        self.sampling = sampling
        self.batch_size = batch_size
        self.opt_map = opt_map
        self.callback_foms = callback_foms
        self.callback_figs = callback_figs
        self.inverse = False
        self.options = options
        self.log_setup(dir_path)

    def log_setup(self, dir_path):
        self.dir_path = dir_path
        self.string = self.algorithm.__name__ + '-' \
                 + self.sampling + '-' \
                 + str(self.batch_size) + '-' \
                 + self.fom.__name__
        # datafile = os.path.basename(self.datafile)
        # datafile = datafile.split('.')[0]
        # string = string + '----[' + datafile + ']'
        self.logdir = log_setup(dir_path, self.string)
        self.logname = 'model_learn.log'

    def read_data(self, datafile):
        with open(datafile, 'rb+') as file:
            data = pickle.load(file)
            self.learn_from = data['seqs_grouped_by_param_set']
            self.gateset_opt_map = data['opt_map']

    def load_best(self, init_point):
        with open(init_point) as init_file:
            best = init_file.readlines()
            best_exp_opt_map = [tuple(a) for a in json.loads(best[0])]
            init_p = json.loads(best[1])['params']
            self.exp.set_parameters(init_p, best_exp_opt_map)
            print("\nLoading previous best point.")

    def select_from_data(self):
        learn_from = self.learn_from
        sampling = self.sampling
        batch_size = self.batch_size
        total_size = len(learn_from)
        all = np.arange(total_size)
        if sampling == 'random':
            indeces = np.random.sample(all, batch_size)
        elif sampling == 'even':
            n = int(total_size / batch_size)
            indeces = all[::n]
        elif sampling == 'from_start':
            indeces = all[:batch_size]
        elif sampling == 'from_end':
            indeces = all[-batch_size:]
        elif sampling == 'all':
            indeces = all
        else:
            raise(
                """Unspecified sampling method.\n
                Select from 'from_end'  'even', 'random' , 'from_start', 'all'.
                Thank you."""
            )
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
        print(f"\nSaving as:\n{self.logdir + self.logname}")
        x0 = self.exp.get_parameters(self.opt_map, scaled=True)
        try:
            # TODO deal with kears learning differently
            if self.grad:
                self.algorithm(
                    x0,
                    self.fct_to_min,
                    self.lookup_gradient
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
        self.exp.set_parameters(best_params, self.opt_map)
        self.end_log()
        self.confirm()

    def confirm(self):
        self.logname = 'confirm.log'
        self.inverse = True
        self.start_log()
        print(f"\nSaving as:\n{self.logdir + self.logname}")
        x_best = self.exp.get_parameters(self.opt_map, scaled=True)
        self.evaluation = -1
        try:
            self.goal_run(x_best)
        except KeyboardInterrupt:
            pass

    def goal_run(self, current_params):
        indeces = self.select_from_data()
        self.exp.set_parameters(current_params, self.opt_map, scaled=True)
        exp_values = []
        exp_stds = []
        sim_values = []

        count = 0
        for ipar in indeces:
            count += 1
            m = self.learn_from[ipar]
            gateset_params = m['params']
            gateset_opt_map = self.gateset_opt_map
            m_vals = m['results']
            m_stds = m['results_std']
            sequences = m['seqs']
            num_seqs = len(sequences)

            self.exp.gateset.set_parameters(
                gateset_params, gateset_opt_map, scaled=False
            )
            sim_vals = self.exp.evaluate(sequences)

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
            for iseq in range(num_seqs):
                m_val = m_vals[iseq]
                m_std = m_stds[iseq]
                exp_values.append(m_val)
                exp_stds.append(m_std)
                sim_val = float(sim_vals[iseq].numpy())
                with open(self.logdir + self.logname, 'a') as logfile:
                    logfile.write(
                        " Sequence {} of {}:\n".format(iseq + 1, num_seqs)
                    )
                    logfile.write(f"  Simulation: {sim_val:8.5f}")
                    logfile.write(f"  Experiment: {m_val:8.5f}")
                    logfile.write(f"  Std: {m_std:8.5f}")
                    logfile.write(f"  Diff: {m_val-sim_val:8.5f}\n")
                    logfile.flush()

        exp_values = tf.constant(exp_values, dtype=tf.float64)
        sim_values = tf.concat(sim_values, axis=0)
        exp_stds = tf.constant(exp_stds, dtype=tf.float64)
        goal = self.fom(exp_values, sim_values, exp_stds)
        goal_numpy = float(goal.numpy())

        with open(self.logdir + self.logname, 'a') as logfile:
            logfile.write("Finished batch with\n")
            logfile.write("{}: {}\n".format(self.fom.__name__, goal_numpy))
            for cb_fom in self.callback_foms:
                val = float(cb_fom(exp_values, sim_values, exp_stds).numpy())
                logfile.write("{}: {}\n".format(cb_fom.__name__, val))
            logfile.flush()

        for cb_fig in self.callback_figs:
            fig = cb_fig(exp_values, sim_values, exp_stds)
            fig.savefig(
                self.logdir
                + cb_fig.__name__ + '/'
                + 'eval:' + str(self.evaluation) + "__"
                + self.fom.__name__ + str(round(goal_numpy, 3))
                + '.png'
            )
            plt.close(fig)
        fig, axs = self.exp.plot_dynamics(self.exp.psi_init, sequences[0])
        l, r = axs.get_xlim()
        axs.plot(r, m_val, 'kx')
        fig.savefig(
            self.logdir
            + 'dynamics_seq/'
            + 'eval:' + str(self.evaluation) + "__"
            + self.fom.__name__ + str(round(goal_numpy, 3))
            + '.png'
        )
        plt.close(fig)
        fig, axs = self.exp.plot_dynamics(
            self.exp.psi_init,
            ['X90p', 'Y90p', 'X90p', 'Y90p']
        )
        fig.savefig(
            self.logdir
            + 'dynamics_xyxy/'
            + 'eval:' + str(self.evaluation) + "__"
            + self.fom.__name__ + str(round(goal_numpy, 3))
            + '.png'
        )
        plt.close(fig)
        display.plot_C3([self.logdir])

        self.optim_status['params'] = [
            par.numpy().tolist()
            for par in self.exp.get_parameters(self.opt_map)
        ]
        self.optim_status['goal'] = goal_numpy
        self.evaluation += 1
        return goal
