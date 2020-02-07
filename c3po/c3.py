"""Object that deals with the model learning."""

import time
import json
import numpy as np
import tensorflow as tf
from c3po.optimizer import Optimizer
from c3po.alorithms import cmaes

class C3(Optimizer):
    """Object that deals with the model learning."""

    def __init__(
        self,
        exp,
        sim,
        eval_func,
        fom,
        data_path,
        learn_from,
        sampling,
        opt_map,
        gateset_opt_map,
        algorithm_no_grad=cmaes,
        algorithm_with_grad=None,
        callback_foms=[],
        opt_name='learn_model',
    ):
        """Initiliase."""
        self.sim = sim
        self.eval_func = eval_func
        self.fom = fom
        self.callback_foms = callback_foms
        self.opt_name = opt_name
        self.data = data_path
        self.logfile_name = self.data_path + self.opt_name + '.log'
        self.optim_status = {}
        self.learn_from = learn_from
        self.sampling = sampling
        self.opt_map = opt_map
        self.gateset_opt_map = gateset_opt_map

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
            logfile.write("Optimization parameters:\n\n")
            logfile.write(json.dumps(self.opt_map))
            logfile.write("\n")

    def learn_model(self):
        print(f"Saving as:\n{self.logfile_name}")
        x0 = self.sim.exp.get_parameters(self.opt_map, scaled=True)

        if self.algorithm_with_grad:
            self.algorithm(
                x0,
                self.fct_to_min_with_grad,
                self.lookup_gradient
            )
        elif self.algorithm_no_grad:
            self.algorithm_no_grad(
                x0,
                self.fct_to_min
            )
        else:
            raise ValueError("You need to pass an algorithm call")

        # TODO deal with kears learning differently
        # if self.algorithm == 'keras-SDG':
        #     vars = tf.Variable(x0)
        #     self.keras_vars = vars
        #     optimizer = tf.keras.optimizers.SGD(learning_rate=0.1)
        #     optimizer.minimize(self.goal_run_n_keras, var_list=[vars])
        #     x_best = vars.numpy()
        #
        # elif self.algorithm == 'keras-Adam':
        #     vars = tf.Variable(x0)
        #     optimizer = tf.keras.optimizers.Adam(learning_rate=0.1)
        #     optimizer.minimize(self.goal_run_n(vars), var_list=[vars])
        #     x_best = vars.numpy()

        self.sim.exp.set_parameters(self.x_best, self.opt_map, scaled=True)
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
        self.sim.exp.set_parameters(current_params, self.opt_map, scaled=True)
        batch_size = len(measurements)

        for ipar in range(len(measurements)):
            m = measurements[ipar]
            gateset_params = m['params']
            sequences = m['seqs']
            num_seqs = len(sequences)
            self.sim.gateset.set_parameters(
                gateset_params, self.gateset_opt_map, scaled=False
            )
            U_dict = self.sim.get_gates()

            with open(self.logfile_name, 'a') as logfile:
                logfile.write(
                    "\n  Parameterset {} of {}:  {}".format(
                        ipar,
                        batch_size,
                        self.sim.gateset.get_parameters(
                            self.gateset_opt_map, to_str=True
                        )
                    )
                )

            exp_values = []
            sim_values = []
            exp_stds = []
            for iseq in range(num_seqs):
                seq = sequences[iseq]
                gates = seq['gate_seq']
                exp_val = seq['result']
                exp_std = seq['result_std']

                sim_val = self.eval_func(U_dict, gates)
                sim_val_numpy = float(sim_val.numpy())

                exp_values.append(exp_val)
                sim_values.append(sim_val)
                exp_stds.append(exp_std)

                with open(self.logfile_name, 'a') as logfile:
                    logfile.write(
                        "\n  Sequence {} of {}:\n  {}\n".format(
                            iseq, num_seqs, gates
                        )
                    )
                    iseq += 1
                    logfile.write(
                        f"  Simulation:  {sim_val_numpy:8.5f}"
                    )
                    logfile.write(
                        f"  Experiment: {exp_val:8.5f}"
                    )
                    logfile.write(
                        f"  Diff: {exp_val-sim_val_numpy:8.5f}\n"
                    )
                    logfile.flush()


            with open(self.logfile_name, 'a') as logfile:
                logfile.write(
                    f"  Mean simulation values: {float(np.mean(sim_values)):8.5f}"
                )
                logfile.write(
                    f" std: {float(np.std(sim_values)):8.5f}\n"
                )
                logfile.write(
                    f"  Mean experiment values: {float(np.mean(exp_values)):8.5f}"
                )
                logfile.write(
                    f" std: {float(np.std(exp_values)):8.5f}\n"
                )
                logfile.flush()

        self.sim.plot_dynamics(self.sim.ket_0, seq)

        exp_values = tf.constant(exp_values, dtype=tf.float64)
        sim_values = tf.concat(sim_values, axis=0)
        exp_stds = tf.constant(exp_stds, dtype=tf.float64)
        goal = self.fom(exp_values, sim_values, exp_stds)
        goal_numpy = float(goal.numpy())

        with open(self.logfile_name, 'a') as logfile:
            logfile.write(
                "Finished batch with {}: {}\n".format(
                    self.fom.__name__,
                    goal_numpy
                )
            )
            for cb_fom in self.callback_foms:
                logfile.write(
                    "Finished batch with {}: {}\n".format(
                        cb_fom.__name__,
                        float(cb_fom(exp_values, sim_values, exp_stds).numpy())
                    )
                )
            logfile.flush()

        self.optim_status['params'] = [
            par.numpy().tolist()
            for par in self.exp.get_parameters(self.opt_map)
        ]
        self.optim_status['goal'] = goal_numpy
        return goal
