"""Object that deals with the model learning."""

import time
import json
import pickle
import numpy as np
import tensorflow as tf
from c3po.optimizer import Optimizer
import matplotlib.pyplot as plt


class C3(Optimizer):
    """Object that deals with the model learning."""

    def __init__(
        self,
        fom,
        dir_path,
        sampling,
        batch_size,
        opt_map,
        callback_foms=[],
        callback_figs=[],
        algorithm_no_grad=None,
        algorithm_with_grad=None,
    ):
        """Initiliase."""
        self.fom = fom
        self.dir_path = dir_path
        self.sampling = sampling
        self.batch_size = batch_size
        self.opt_map = opt_map
        self.callback_foms = callback_foms
        self.callback_figs = callback_figs
        super().__init__()

        if algorithm_with_grad:
            self.algorithm = algorithm_with_grad
            self.grad = True
        elif algorithm_no_grad:
            self.algorithm = algorithm_no_grad
            self.grad = False
        else:
            raise ValueError("You need to pass an algorithm call")

        self.logfile_name = self.dir_path + 'learn_model.log'
        self.optim_status = {}

    def set_exp(self, exp):
        self.exp = exp

    def read_data(self, datafile):
        with open(datafile, 'rb+') as file:
            self.learn_from = pickle.load(file)

    def select_from_data(self):
        learn_from = self.learn_from
        sampling = self.sampling
        if sampling == 'random':
            measurements = np.random.sample(learn_from, self.batch_size)
        elif sampling == 'even':
            n = int(len(learn_from) / self.batch_size)
            measurements = learn_from[::n]
        elif sampling == 'from_start':
            measurements = learn_from[:self.batch_size]
        elif sampling == 'from_end':
            measurements = learn_from[-self.batch_size:]
        elif sampling == 'ALL':
            measurements = learn_from
        else:
            raise(
                """Unspecified sampling method.\n
                Select from 'from_end'  'even', 'random' , 'from_start', 'ALL'.
                Thank you."""
            )
        return measurements  # list(set(learn_from) - set(measurements))

    def start_log(self):
        self.start_time = time.time()
        start_time_str = str(f"{time.asctime(time.localtime())}\n\n")
        with open(self.logfile_name, 'a') as logfile:
            logfile.write("Starting optimization at ")
            logfile.write(start_time_str)
            logfile.write("Optimization parameters:\n")
            logfile.write(json.dumps(self.opt_map))
            logfile.write("\n")

    def learn_model(self):
        print(f"Saving as:\n{self.logfile_name}")
        x0 = self.exp.get_parameters(self.opt_map, scaled=True)
        if self.grad:
            self.algorithm(
                x0,
                self.fct_to_min_with_grad,
                self.lookup_gradient
            )
        elif self.grad:
            self.algorithm(
                x0,
                self.fct_to_min
            )
        else:
            raise ValueError("You need to pass an algorithm call")

        # TODO deal with kears learning differently
        self.exp.set_parameters(self.x_best, self.opt_map, scaled=True)
        self.end_time = time.time()
        with open(self.logfile_name, 'a') as logfile:
            logfile.write(
                f"Finished at {time.asctime(time.localtime())}\n"
            )
            logfile.write(
                f"Total runtime: {self.end_time-self.start_time}\n\n"
            )
            logfile.flush()

    def fct_to_min(self, x):
        current_params = tf.constant(x)
        measurements = self.select_from_data(self.sampling)
        goal = self.goal_run_n(current_params, measurements)
        self.log_parameters()
        return float(goal.numpy())

    def fct_to_min_with_grad(self, x):
        current_params = tf.constant(x)
        measurements = self.select_from_data(self.sampling)
        goal = self.goal_run_n_with_grad(current_params, measurements)
        self.log_parameters()
        return float(goal.numpy())

    def goal_run_n_with_grad(self, current_params, measurements):
        with tf.GradientTape() as t:
            t.watch(current_params)
            goal = self.goal_run_n(self, current_params, measurements)
        grad = t.gradient(goal, current_params)
        gradients = grad.numpy().flatten()
        self.gradients[str(current_params.numpy())] = gradients
        self.optim_status['gradient'] = gradients.tolist()

    def goal_run_n(self, current_params, measurements):
        self.exp.set_parameters(current_params, self.opt_map, scaled=True)
        batch_size = len(measurements)
        exp_values = []
        exp_stds = []
        sim_values = []

        for ipar in range(len(measurements)):
            m = measurements[ipar]
            gateset_params = m['params']
            gateset__opt_map = m["opt_map"]
            sequences = m['seqs']
            m_vals = m['results']
            m_stds = m['result_stds']
            num_seqs = len(sequences)

            self.exp.gateset.set_parameters(
                gateset_params, gateset__opt_map, scaled=False
            )
            sim_vals = self.exp.evaluate()

            exp_values.extend(m_vals)
            exp_stds.extend(m_stds)
            sim_values.extend(sim_vals)

            with open(self.logfile_name, 'a') as logfile:
                logfile.write(
                    "\n  Parameterset {} of {}:\n {}\n {}".format(
                        ipar,
                        batch_size,
                        logfile.write(json.dumps(self.opt_map)),
                        self.sim.gateset.get_parameters(
                            self.gateset_opt_map, to_str=True
                        ),
                    )
                )
            for iseq in range(num_seqs):
                m_val = m_vals[iseq]
                m_std = m_stds[iseq]
                sim_val = sim_vals[iseq].numpy()
                with open(self.logfile_name, 'a') as logfile:
                    logfile.write(
                        " Sequence {} of {}:\n".format(iseq, num_seqs)
                    )
                    logfile.write(f" Simulation:  {sim_val:8.5f}")
                    logfile.write(f" Experiment: {m_val:8.5f}")
                    logfile.write(f" Std: {m_std:8.5f}")
                    logfile.write(f" Diff: {m_val-sim_val:8.5f}\n")
                    logfile.flush()

        exp_values = tf.constant(exp_values, dtype=tf.float64)
        sim_values = tf.concat(sim_values, axis=0)
        exp_stds = tf.constant(exp_stds, dtype=tf.float64)
        goal = self.fom(exp_values, sim_values, exp_stds)
        goal_numpy = float(goal.numpy())

        with open(self.logfile_name, 'a') as logfile:
            logfile.write("Finished batch with\n")
            logfile.write("{}: {}\n".format(self.fom.__name__, goal_numpy))
            for cb_fom in self.callback_foms:
                val = float(cb_fom(exp_values, sim_values, exp_stds).numpy())
                logfile.write("{}: {}\n".format(cb_fom.__name__, val))
            logfile.flush()

        for cb_fig in self.callback_figs:
            fig = cb_fig(exp_values, sim_values, exp_stds)
            fig.savefig(
                self.dir_path
                + cb_fig.__name__ + '/'
                + 'eval:' + str(self.evaluation) + "__"
                + self.fom.__name__ + str(round(goal.numpy(), 3))
                + '.png'
            )
            plt.close(fig)
        fig, axs = self.sim.plot_dynamics(self.sim.ket_0, sequences[0])
        l, r = axs.get_xlim()
        axs.plot(r, fid, 'x')
        fig.savefig(
            self.dir_path
            + 'dynamics_seq/'
            + 'eval:' + str(self.evaluation) + "__"
            + self.fom.__name__ + str(round(goal.numpy(), 3))
            + '.png'
        )
        plt.close(fig)
        fig, axs = self.sim.plot_dynamics(
            self.sim.ket_0,
            ['X90p','Y90p','X90p','Y90p']
        )
        fig.savefig(
            self.dir_path
            + 'dynamics_xyxy/'
            + 'eval:' + str(self.evaluation) + "__"
            + self.fom.__name__ + str(round(goal.numpy(), 3))
            + '.png'
        )
        plt.close(fig)
        c3po.display.plot_learning(self.dir_path)

        self.optim_status['params'] = [
            par.numpy().tolist()
            for par in self.exp.get_parameters(self.opt_map)
        ]
        self.optim_status['goal'] = goal_numpy
        return goal

    def log_parameters(self):
        if self.optim_status['goal'] < self.current_best_goal:
            self.current_best_goal = self.optim_status['goal']
            with open(self.dir_path+'best_point', 'w') as best_point:
                best_point.write(json.dumps(self.opt_map))
                best_point.write("\n")
                best_point.write(json.dumps(self.optim_status))
        self.logfile.write(json.dumps(self.optim_status))
        self.logfile.write("\n")
        self.logfile.write(f"\nFinished evaluation {self.evaluation}\n")
        self.evaluation += 1
        self.logfile.flush()
