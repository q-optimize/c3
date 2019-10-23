"""Optimizer object, where the optimal control is done."""

import pickle
import random
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

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
        self.shai_fid = False
        plt.rcParams['figure.dpi'] = 100

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
            current_params = tf.constant(
                self.to_bound_phys_scale(x, self.bounds)
            )
            t.watch(current_params)
            goal = 0
            batch_size = 20

            if self.random_samples:
                measurements = random.sample(
                    self.optimizer_logs[learn_from], batch_size
                    )
            else:
                measurements = self.optimizer_logs[learn_from][-batch_size::]
            for m in measurements:
                this_goal = self.eval_func(
                    current_params, self.opt_map,  m[0], m[1]
                    )
                self.optimizer_logs['per_point_error'].append(
                    float(this_goal.numpy())
                    )
                goal += this_goal

            if self.shai_fid:
                goal = np.log10(np.sqrt(goal/batch_size))

        grad = t.gradient(goal, current_params)
        self.gradients[str(x)] = grad.numpy().flatten()
        self.optimizer_logs[self.optim_name].append(
            [current_params, float(goal.numpy())]
            )

        return goal.numpy()

    def goal_gradient_run(self, x):
        grad = self.gradients[str(x)]
        scale = np.diff(self.bounds)
        return grad*scale.T

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
                    goal = (1+0.03*np.random.randn()) * goal

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

        # cmaes res is tuple, tread carefully
        # res[0] = values_opt

        self.results[self.optim_name] = res

        return values_opt

    def lbfgs(self, values, bounds, goal, grad, settings={}):
        x0 = self.to_scale_one(values, bounds)
        settings['disp'] = True
        res = minimize(
                goal,
                x0,
                jac=grad,
                method='L-BFGS-B',
                options=settings,
                callback=self.plot_progress
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
        controls : class ControlSet
            control Class carrying all relevant information

        opt_map : list

        opt : type
            Specification of the optimizer to be used, i.e. cmaes, powell, ...

        settings : dict
            Special settings for the desired optimizer

        """
        fig, axs = plt.subplots(1, 1)
        self.fig = fig
        self.axs = axs

        values, bounds = controls.get_corresponding_control_parameters(opt_map)
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
            values_opt = self.cmaes(values, bounds, settings)

        elif opt == 'lbfgs':
            values_opt = self.lbfgs(
                values,
                bounds,
                self.goal_run,
                self.goal_gradient_run,
                settings=settings
            )

        values = values_opt
        # TODO make these two happend at the same time if you pass a save name
        # TODO use update_controls here, (and change to set_values)
        controls.update_controls(values, opt_map)
        controls.save_params_to_history(calib_name)
        self.parameter_history[calib_name] = values

    def learn_model(
        self,
        model,
        eval_func,
        settings,
        learn_from,
        optim_name='learn_model',
        ):

        fig, axs = plt.subplots(1, 1)
        self.fig = fig
        self.axs = axs

        values, bounds = model.get_values_bounds()
        bounds = np.array(bounds)
        self.bounds = bounds
        values = np.array(values)
        self.param_shape = values.shape
        self.eval_func = eval_func

        self.learn_from = learn_from
        self.optim_name = optim_name
        self.optimizer_logs[self.optim_name] = []
        self.optimizer_logs['per_point_error'] = []
        params_opt = self.lbfgs(
                    values,
                    bounds,
                    self.goal_run_n,
                    self.goal_gradient_run,
                    settings=settings
                )
        model.params = np.array(params_opt)

    def plot_progress(self, res=None):
        fig = self.fig
        ax = self.axs
        self.goal.append(self.optimizer_logs[self.optim_name][-1][1])
        ax.clear()
        ax.semilogy(self.goal)
        ax.set_title(self.optim_name)
        ax.set_xlabel('Iteration')
        ax.set_ylabel('1-Fidelitiy')
        ax.grid()
        fig.canvas.draw()
        fig.canvas.flush_events()
