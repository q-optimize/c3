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

    def __init__(self, data_path):
        self.results = {}
        self.gradients = {}
        self.simulate_noise = False
        self.sampling = False
        self.batch_size = 1
        self.data_path = data_path

    def to_scale_one(self, values, bounds):
        """
        Return a vector of scale 1 that plays well with optimizers.

        If the input is higher dimension, it also linearizes.

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
        # TODO Give warning when outside bounds
        x0 = []
        values = values.flatten()
        for i in range(len(values)):
            scale = np.abs(bounds[i].T[0] - bounds[i].T[1])
            offset = min(bounds[i].T)
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
            offset = min(bounds[i].T)
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
            U_dict = self.sim.get_gates(current_params, self.opt_map)
            goal = self.fid_func(U_dict)

        grad = t.gradient(goal, current_params)
        scale = np.diff(self.bounds)
        gradients = grad.numpy().flatten() * scale.T
        self.gradients[str(x)] = gradients
        self.optim_status['params'] = list(zip(
            self.opt_map, current_params.numpy()
        ))
        self.optim_status['goal'] = float(goal.numpy())
        self.optim_status['gradient'] = gradients.tolist()
        return float(goal.numpy())

    def goal_run_n(self, x):
        learn_from = self.learn_from
        with tf.GradientTape() as t:
            current_params = tf.constant(
                self.to_bound_phys_scale(x, self.bounds)
            )
            t.watch(current_params)
            goal = 0

            if self.sampling == 'random':
                measurements = random.sample(learn_from, self.batch_size)
            elif self.sampling == 'even':
                n = int(len(learn_from) / self.batch_size)
                measurements = learn_from[-n:]
            elif self.sampling == 'from_start':
                measurements = learn_from[:self.batch_size]
            elif self.sampling == 'from_end':
                measurements = learn_from[-self.batch_size:]
            batch_size = len(measurements)
            for m in measurements:
                gateset_params = m[0]
                seq = m[1]
                fid = m[2]
                this_goal = self.eval_func(
                    current_params,
                    self.opt_map,
                    gateset_params,
                    self.gateset_opt_map,
                    seq,
                    fid
                )
                self.logfile.write(
                    f"\n  Sequence:  {seq}\n"
                )
                self.logfile.write(
                    f"  Simulation:  {float(this_goal.numpy())+fid:8.5f}"
                )
                self.logfile.write(
                    f"  Experiment: {fid:8.5f}"
                )
                self.logfile.write(
                    f"  Diff: {float(this_goal.numpy()):8.5f}\n"
                )
                self.logfile.flush()
                goal += this_goal ** 2

            goal = tf.sqrt(goal / batch_size)
            self.logfile.write(
                f"Finished batch with RMS: {float(goal.numpy())}\n"
            )
            self.logfile.flush()

        grad = t.gradient(goal, current_params)
        scale = np.diff(self.bounds)
        gradients = grad.numpy().flatten() * scale.T
        self.gradients[str(x)] = gradients
        self.optim_status['params'] = list(zip(
            self.opt_map, current_params.numpy()
        ))
        self.optim_status['goal'] = float(goal.numpy())
        self.optim_status['gradient'] = gradients.tolist()
        return goal.numpy()

    def goal_gradient_run(self, x):
        return self.gradients[str(x)]

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
                self.logfile.write(
                    json.dumps({
                        'params': self.to_bound_phys_scale(
                            sample, bounds
                        ).tolist(),
                        'goal': goal})
                )
                self.logfile.write("\n")
                self.logfile.flush()
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
        self.results[self.opt_name] = res
        return values_opt

    def lbfgs(self, values, bounds, goal, grad):
        x0 = self.to_scale_one(values, bounds)
        self.optim_status['params'] = list(zip(
            self.opt_map, values
        ))
        self.log_parameters(x0)
        res = minimize(
            goal,
            x0,
            jac=grad,
            method='L-BFGS-B',
            callback=self.log_parameters,
            options={'disp': True}
        )

        values_opt = self.to_bound_phys_scale(res.x, bounds)
        res.x = values_opt
        self.results[self.opt_name] = res
        return values_opt

    def optimize_controls(
        self,
        sim,
        opt_map,
        opt,
        opt_name,
        fid_func,
        opt_settings={}
    ):
        """
        Apply a search algorightm to your gateset given a fidelity function.

        Parameters
        ----------
        simulator : class Simulator
            simulator class carrying all relevant informatioFn:
            - experiment object with model and generator
            - gateset object with pulse information for each gate

        opt_map : list
            Specifies which parameters will be optimized

        opt : str
            Specification of the optimizer to be used, i.e. cmaes, powell, ...

        calib_name : str

        fid_func : function
            Takes the dictionary of gates and outputs a fidelity value of which
            we want to find the minimum

        settings : dict
            Special settings for the desired optimizer

        """

        values, bounds = sim.gateset.get_parameters(opt_map)
        self.param_shape = values.shape
        self.bounds = bounds
        self.sim = sim
        self.opt_map = opt_map
        self.opt_name = opt_name
        self.fid_func = fid_func
        self.optim_status = {}
        self.iteration = 1

        # TODO fix this horrible mess and make the shape of bounds general for
        # PWC and carrier based controls
        if len(self.param_shape) > 1:
            bounds = bounds.reshape(bounds.T.shape)

        self.logfile_name = self.data_path + self.opt_name + '.log'
        print(f"Saving as:\n{self.logfile_name}")
        start_time = time.time()
        with open(self.logfile_name, 'w') as self.logfile:
            self.logfile.write(
                f"Starting optimization at {time.asctime(time.localtime())}\n\n"
            )
            self.logfile.flush()
            if opt == 'cmaes':
                values_opt = self.cmaes(
                    values,
                    self.bounds,
                    opt_settings
                )

            elif opt == 'lbfgs':
                values_opt = self.lbfgs(
                    values,
                    self.bounds,
                    self.goal_run,
                    self.goal_gradient_run
                )
            end_time = time.time()
            self.logfile.write(
                f"Finished at {time.asctime(time.localtime())}\n"
            )
            self.logfile.write(
                f"Total runtime:{end_time-start_time}"
            )
            self.logfile.flush()

        sim.gateset.set_parameters(values_opt, opt_map)
        # TODO decide if gateset object should have history and implement
        # TODO save while setting if you pass a save name
        # pseudocode: controls.save_params_to_history(calib_name)

    def learn_model(
        self,
        exp,
        eval_func,
        opt_name='learn_model',
        settings={}
    ):
        # TODO allow for specific data from optimizer to be used for learning
        values, bounds = exp.get_parameters(self.opt_map)
        bounds = np.array(bounds)
        self.bounds = bounds
        values = np.array(values)
        self.param_shape = values.shape
        self.eval_func = eval_func

        self.opt_name = opt_name
        self.logfile_name = self.data_path + self.opt_name + '.log'
        print(f"Saving as:\n{self.logfile_name}")
        self.optim_status = {}
        self.iteration = 0

        with open(self.logfile_name, 'w') as self.logfile:
            start_time = time.time()
            self.logfile.write(
                f"Starting optimization at {time.asctime(time.localtime())}\n\n"
            )
            params_opt = self.lbfgs(
                values,
                bounds,
                self.goal_run_n,
                self.goal_gradient_run
            )
            end_time = time.time()
            self.logfile.write(
                f"Finished at {time.asctime(time.localtime())}\n"
            )
            self.logfile.write(
                f"Total runtime: {end_time-start_time}"
            )
            self.logfile.flush()
        exp.set_parameters(params_opt, self.opt_map)

    def log_parameters(self, x):
        self.logfile.write(json.dumps(self.optim_status))
        self.logfile.write("\n")
        self.logfile.write(f"\nStarting iteration {self.iteration}\n")
        self.iteration += 1
        self.logfile.flush()
