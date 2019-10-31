"""Optimizer object, where the optimal control is done."""

import pickle
import random
import os
import time
import json
import numpy as np
import tensorflow as tf

from scipy.optimize import minimize as minimize
import cma.evolution_strategy as cmaes


class Optimizer:
    """Optimizer object, where the optimal control is done."""

    def __init__(self):
        self.optimizer_logs = {}
        self.parameter_history = {}
        self.results = {}
        self.gradients = {}
        self.simulate_noise = False
        self.random_samples = False
        self.batch_size = 1
        self.shai_fid = False

    def save_history(self, filename):
        datafile = open(filename, 'wb')
        pickle.dump(self.optimizer_logs, datafile)
        datafile.close()
        pass

    def load_history(self, filename):
        file = open(filename, 'rb')
        self.optimizer_logs = pickle.load(file)
        file.close()
        pass

    def set_session(self, sess):
        self.sess = sess

    def set_log_writer(self, writer):
        self.log_writer = writer

    def to_scale_one(self, values, bounds):
        """
        Return a vector of scale 1 that plays well with optimizers.

        If theinput is higher dimension, it also linearizes.

        Parameters
        ----------
        values : array/str
            Array of parameter in physical units. Can also be the name of an
            array already stored in this Gate instance.

        Returns
        -------
        array
            Numpy array of pulse parameters, rescaled to values within [-1, 1]

        """
        x0 = []
        values = values.flatten()
        for i in range(len(values)):
            scale = np.abs(bounds[i].T[0] - bounds[i].T[1])
            offset = bounds[i].T[0]

            tmp = (values[i] - offset) / scale
            tmp = 2 * tmp - 1
            x0.append(tmp)

        return x0

    def to_bound_phys_scale(self, x0, bounds):
        """
        Transform an optimizer vector back to physical scale & original shape.

        Parameters
        ----------
        one : array
            Array of pulse parameters in scale 1

        bounds: array
            Array of control parameter bounds

        Returns
        -------
        array
            control parameters, compatible with bounds in physical units.

        """
        values = []
        for i in range(len(x0)):
            scale = np.abs(bounds[i].T[0] - bounds[i].T[1])
            offset = bounds[i].T[0]
            tmp = np.arccos(np.cos((x0[i] + 1) * np.pi / 2)) / np.pi
            tmp = scale * tmp + offset
            values.append(tmp)
        return np.array(values).reshape(self.param_shape)

    def goal_run(self, x):
        with tf.GradientTape() as t:
            current_params = tf.constant(
                self.to_bound_phys_scale(x, self.bounds)
            )
            t.watch(current_params)
            goal = self.eval_func(current_params, self.opt_map)

        grad = t.gradient(goal, current_params)
        self.gradients[str(x)] = grad.numpy().flatten()
        # print(goal)
        # print(goal.numpy())
        self.optimizer_logs[self.optim_name].append(
            [current_params, float(goal.numpy())]
        )
        return float(goal.numpy())

    def goal_run_n(self, x):
        learn_from = self.learn_from
        with tf.GradientTape() as t:
            exp_params = tf.constant(
                self.to_bound_phys_scale(x, self.bounds)
            )
            t.watch(exp_params)
            goal = 0
            batch_size = self.batch_size

            if self.random_samples:
                measurements = random.sample(learn_from, batch_size)
            else:
                measurements = learn_from[-batch_size::]
            for m in measurements:
                gateset_params = m[0]
                seq = m[1]
                fid = m[2]
                this_goal = self.eval_func(
                    exp_params,
                    self.exp_opt_map,
                    gateset_params,
                    self.gateset_opt_map,
                    seq,
                    fid
                )
                self.logfile.write(
                    f"  Simulation:  {abs(float(this_goal.numpy())-fid):8.5f}"
                )
                self.logfile.write(
                    f"  Experiment: {fid:8.5f}"
                )
                self.logfile.write(
                    f"  Error: {float(this_goal.numpy()):8.5f}\n"
                )
                self.logfile.flush()
                self.optimizer_logs['per_point_error'].append(
                    float(this_goal.numpy())
                )
                goal += this_goal

            goal = tf.sqrt(goal / batch_size)
            self.goal.append(goal)

            if self.shai_fid:
                goal = np.log10(np.sqrt(goal / batch_size))

        grad = t.gradient(goal, exp_params)
        self.gradients[str(x)] = grad.numpy().flatten()
        self.optimizer_logs[self.optim_name].append(
            [exp_params, float(goal.numpy())]
        )
        self.optim_status['params'] = list(zip(
            self.exp_opt_map, exp_params.numpy()
        ))
        self.optim_status['goal'] = float(goal.numpy())
        self.optim_status['gradient'] = list(grad.numpy())
        return goal.numpy()

    def goal_gradient_run(self, x):
        grad = self.gradients[str(x)]
        scale = np.diff(self.bounds)
        return grad * scale.T

    def cmaes(self, values, bounds, settings={}):
        # TODO: rewrite from dict to list input
        if settings:
            if 'CMA_stds' in settings.keys():
                scale_bounds = []
                for i in range(len(bounds)):
                    scale = np.abs(bounds[i][0] - bounds[i][1])
                    scale_bounds.append(settings['CMA_stds'][i] / scale)
                settings['CMA_stds'] = scale_bounds

        x0 = self.to_scale_one(values, bounds)
        es = cmaes.CMAEvolutionStrategy(x0, 1, settings)
        while not es.stop():
            samples = es.ask()
            solutions = []
            for sample in samples:
                goal = float(self.goal_run(sample))
                if self.simulate_noise:
                    goal = (1 + 0.03 * np.random.randn()) * goal
                solutions.append(goal)
                self.plot_progress()
            es.tell(
                    samples,
                    solutions
                    )
            es.logger.add()
            es.disp()

        res = es.result + (es.stop(), es, es.logger)
        x_opt = res[0]
        values_opt = self.to_bound_phys_scale(x_opt, bounds)
        # cmaes res is tuple, tread carefully. res[0] = values_opt
        self.results[self.optim_name] = res
        return values_opt

    def lbfgs(self, values, bounds, goal, grad, settings={}):
        x0 = self.to_scale_one(values, bounds)
        res = minimize(
            goal,
            x0,
            jac=grad,
            method='L-BFGS-B',
            callback=self.log_parameters
        )

        values_opt = self.to_bound_phys_scale(res.x, bounds)
        res.x = values_opt
        self.results[self.optim_name] = res
        return values_opt

    def optimize_controls(
        self,
        controls,
        opt_map,
        opt,
        calib_name,
        eval_func,
        settings={},
        callback=None
    ):
        """
        Apply a search algorightm to your parameters given a fidelity.

        Parameters
        ----------
        controls : class GateSet
            control Class carrying all relevant information

        opt_map : list

        opt : type
            Specification of the optimizer to be used, i.e. cmaes, powell, ...

        settings : dict
            Special settings for the desired optimizer

        """
        values, bounds = controls.get_parameters(opt_map)
        self.opt_map = opt_map
        self.optim_name = calib_name

        self.goal = []

        self.callback = callback
        # TODO Make sure values and bounds are already np.arrays
        values = np.array(values)
        self.param_shape = values.shape
        bounds = np.array(bounds)
        # TODO fix this horrible mess and make the shape of bounds general for
        # PWC and carrier based controls
        if len(self.param_shape) > 1:
            bounds = bounds.reshape(bounds.T.shape)
        self.bounds = bounds
        self.eval_func = eval_func

        self.optimizer_logs[self.optim_name] = []

        if opt == 'cmaes':
            values_opt = self.cmaes(
                values,
                bounds,
                settings
            )

        elif opt == 'lbfgs':
            values_opt = self.lbfgs(
                values,
                bounds,
                self.goal_run,
                self.goal_gradient_run,
                settings=settings
            )
        controls.set_parameters(values_opt, opt_map)
        # TODO decide if gateset object should have history and implement
        # TODO save while setting if you pass a save name
        # pseudocode: controls.save_params_to_history(calib_name)
        self.parameter_history[calib_name] = values_opt

    def learn_model(
        self,
        exp,
        eval_func,
        optim_name='learn_model',
        settings={}
    ):
        # TODO allow for specific data from optimizer to be used for learning
        values, bounds = exp.get_parameters(self.exp_opt_map)
        bounds = np.array(bounds)
        self.bounds = bounds
        values = np.array(values)
        self.param_shape = values.shape
        self.eval_func = eval_func

        self.optim_name = optim_name
        self.optimizer_logs[self.optim_name] = []
        self.optimizer_logs['per_point_error'] = []
        self.goal = []
        self.optim_status = {}
        self.iteration = 0

        self.log_setup()
        with open(self.log_filename, 'w') as self.logfile:
            start_time = time.time()
            self.logfile.write(
                f"Starting optimization at {time.asctime(time.localtime())}\n\n"
            )
            params_opt = self.lbfgs(
                values,
                bounds,
                self.goal_run_n,
                self.goal_gradient_run,
                settings=settings
            )
            end_time = time.time()
            self.logfile.write(
                f"Finished at {time.asctime(time.localtime())}\n"
            )
            self.logfile.write(
                f"Total runtime:{end_time-start_time}"
            )
        exp.set_parameters(params_opt, self.exp_opt_map)

    def log_setup(self):
        data_path = "/tmp/c3logs/"
        if not os.path.isdir(data_path):
            os.makedirs(data_path)
        pwd = data_path + time.strftime(
            "%Y%m%d%H%M%S", time.localtime()
        )
        os.makedirs(pwd)
        self.log_filename = pwd + '/' + self.optim_name + ".log"
        print(f"Saving to:\n {self.log_filename}\n")

    def log_parameters(self, x):
        self.logfile.write("\n")
        self.logfile.write(json.dumps(self.optim_status))
        self.logfile.write("\n")
        self.iteration += 1
        self.logfile.write(f"Starting iteration {self.iteration}\n")
        self.logfile.flush()
