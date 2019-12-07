"""Optimizer object, where the optimal control is done."""

import os
import random
import time
import json
import numpy as np
import tensorflow as tf

from scipy.optimize import minimize as minimize
import cma.evolution_strategy as cmaes


class Optimizer:
    """Optimizer object, where the optimal control is done."""

    def __init__(self, data_path):
        self.gradients = {}
        self.noise_level = 0
        self.sampling = False
        self.batch_size = 1
        self.data_path = data_path
        self.skip_bad_points = False  # The Millikan option

    def goal_run(self, x_in):
        self.sim.gateset.set_parameters(x_in, self.opt_map, scaled=True)
        U_dict = self.sim.get_gates()
        goal = self.fid_func(U_dict)
        self.optim_status['params'] = list(zip(
            self.opt_map, self.sim.gateset.get_parameters(self.opt_map)
        ))
        self.optim_status['goal'] = float(goal.numpy())
        return float(goal.numpy())

    def goal_run_with_grad(self, x_in):
        with tf.GradientTape() as t:
            x = tf.constant(x_in)
            t.watch(x)
            self.sim.gateset.set_parameters(x, self.opt_map, scaled=True)
            U_dict = self.sim.get_gates()
            goal = self.fid_func(U_dict)

        grad = t.gradient(goal, x)
        gradients = grad.numpy().flatten()
        self.gradients[str(x_in)] = gradients
        self.optim_status['params'] = list(zip(
            self.opt_map, self.sim.gateset.get_parameters(self.opt_map)
        ))
        self.optim_status['goal'] = float(goal.numpy())
        self.optim_status['gradient'] = gradients.tolist()
        return float(goal.numpy())

    def goal_run_n(self, x_in):
        learn_from = self.learn_from
        with tf.GradientTape() as t:
            current_params = tf.constant(x_in)
            t.watch(current_params)
            self.exp.set_parameters(current_params, self.opt_map)

            if self.sampling == 'random':
                measurements = random.sample(learn_from, self.batch_size)
            elif self.sampling == 'even':
                n = int(len(learn_from) / self.batch_size)
                measurements = learn_from[::n]
            elif self.sampling == 'from_start':
                measurements = learn_from[:self.batch_size]
            elif self.sampling == 'from_end':
                measurements = learn_from[-self.batch_size:]
            else:
                raise(
                    """Unspecified sampling method.\n
                    Select from 'from_end'  'even', 'random' , 'from_start'.
                    Thank you."""
                )
            batch_size = len(measurements)
            ipar = 1
            goal = 0
            used_seqs = 0
            for m in measurements:
                gateset_params = m[0]
                self.sim.gateset.set_parameters(
                    gateset_params, self.gateset_opt_map
                )
                self.logfile.write(
                    f"\n  Parameterset {ipar} of {batch_size}:  {self.sim.gateset.get_parameters(self.gateset_opt_map, to_str=True)}\n"
                )
                ipar += 1
                U_dict = self.sim.get_gates()
                iseq = 1
                for seqs in m[1]:
                    seq = seqs[0]
                    fid = seqs[1]
                    if (self.skip_bad_points and fid > 0.25):
                        self.logfile.write(
                            f"\n  Skipped point with infidelity>0.25.\n"
                        )
                        iseq += 1
                        continue
                    this_goal = self.eval_func(U_dict, seq, fid)
                    self.logfile.write(
                        f"\n  Sequence {iseq} of {len(m[1])}:\n  {seq}\n"
                    )
                    iseq += 1
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
                    used_seqs += 1

            goal = tf.sqrt(goal / used_seqs)
            self.logfile.write(
                f"Finished batch with RMS: {float(goal.numpy())}\n"
            )
            self.logfile.flush()

        grad = t.gradient(goal, current_params)
        gradients = grad.numpy().flatten()
        self.gradients[str(x_in)] = gradients
        self.optim_status['params'] = list(zip(
            self.opt_map, self.exp.get_parameters(self.opt_map)
        ))
        self.optim_status['goal'] = float(goal.numpy())
        self.optim_status['gradient'] = gradients.tolist()
        return float(goal.numpy()), gradients

    def lookup_gradient(self, x):
        key = str(x)
        if key not in self.gradients.keys():
            self.goal_run_with_grad(x)
        return self.gradients[key]

    def cmaes(self, x0, settings={}):
        es = cmaes.CMAEvolutionStrategy(x0, 1, settings)
        while not es.stop():
            self.logfile.write(f"Batch {self.iteration}\n")
            self.logfile.flush()
            samples = es.ask()
            solutions = []
            for sample in samples:
                goal = float(self.goal_run(sample))
                goal = (1 + self.noise_level * np.random.randn()) * goal
                solutions.append(goal)
                self.logfile.write(
                    json.dumps({
                        'params': self.sim.gateset.get_parameters(
                                self.opt_map
                                ),
                        'goal': goal})
                )
                self.logfile.write("\n")
                self.logfile.flush()
            self.iteration += 1
            es.tell(
                samples,
                solutions
            )
            es.logger.add()
            es.disp()

        res = es.result + (es.stop(), es, es.logger)
        self.sim.gateset.set_parameters(res[0], self.opt_map, scaled=True)

# TODO desing change? make simulator / optimizer communicate with ask and tell?
    def lbfgs(self, x0, goal, options):
        options['disp'] = True
        res = minimize(
            goal,
            x0,
            jac=self.lookup_gradient,
            method='L-BFGS-B',
            callback=self.log_parameters,
            options=options
        )
        return res.x

    # Adam.
    # Adapted from: https://gluon.mxnet.io/chapter06_optimization/adam-scratch.html

    def Adam_update(
        self, p, grad, vs, sqrs, k_iter, alpha=0.1, beta1 = 0.9, beta2 = 0.999,
        eps_stable = 1e-8
    ):
        if k_iter == 0:
            vs   = eps_stable * np.ones(len(p))
            sqrs = eps_stable * np.ones(len(p))

        for k in range(len(p)):
            vs[k]   = beta1 * vs[k]    + (1 - beta1) * grad[k]
            sqrs[k] = beta2 * sqrs[k]  + (1 - beta2) * np.square(grad[k])

            v_bias_corr   = vs[k]   / (1 - beta1 ** (k_iter+1))
            # Here we want to count from 1
            sqr_bias_corr = sqrs[k] / (1 - beta2 ** (k_iter+1))
            # but Python natively counts form 0

            div  = v_bias_corr / (np.sqrt(sqr_bias_corr) + eps_stable)
            p[k] = p[k] - alpha * div

        return p, vs, sqrs


    def Adam(
        self, loss_and_grad_func, p0, fun_goal=0.03, alpha=0.1,
        beta1 = 0.9, beta2 = 0.999, eps_stable = 1e-8, stopping_func=None
    ):

        p = p0
        vs = []
        sqrs = []
        loss = 1
        k_iter = 0
        while loss > fun_goal:
            loss, loss_grad = loss_and_grad_func(p)

            if stopping_func is not None:
                if stopping_func(k_iter, p, loss, loss_grad):
                    return p

            new_p, vs, sqrs = self.Adam_update(
                p, loss_grad, vs, sqrs, k_iter, alpha, beta1, beta2, eps_stable
            )
            p = new_p
            print(
            f"\nAt iterate    {k_iter}    f=  {loss:E}    g=  {np.linalg.norm(loss_grad):E}\n"
            )
            self.log_parameters(p)
            k_iter += 1

        return p

    def optimize_controls(
        self,
        sim,
        opt_map,
        opt,
        opt_name,
        fid_func,
        settings={}
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
        # TODO Separate gateset from the simulation here.
        x0 = sim.gateset.get_parameters(opt_map, scaled=True)
        self.sim = sim
        self.opt_map = opt_map
        self.opt_name = opt_name
        self.fid_func = fid_func
        self.optim_status = {}
        self.iteration = 1

        # TODO log physical values, not tf values

        self.logfile_name = self.data_path + self.opt_name + '.log'
        print(f"Saving as:\n{self.logfile_name}")
        start_time = time.time()
        with open(self.logfile_name, 'a') as self.logfile:
            start_time_str = str(f"{time.asctime(time.localtime())}\n\n")
            self.logfile.write("Starting optimization at")
            self.logfile.write(start_time_str)
            self.logfile.flush()
            if opt == 'cmaes':
                self.cmaes(
                    x0,
                    settings
                )

            elif opt == 'lbfgs':
                x_best = self.lbfgs(
                    x0,
                    self.goal_run_with_grad,
                    options=settings
                )
                self.sim.gateset.set_parameters(
                    x_best, self.opt_map, scaled=True
                )
            end_time = time.time()
            self.logfile.write("Started at ")
            self.logfile.write(start_time_str)
            self.logfile.write(
                f"Finished at {time.asctime(time.localtime())}\n"
            )
            self.logfile.write(
                f"Total runtime:{end_time-start_time}\n\n"
            )
            self.logfile.flush()

        # TODO decide if gateset object should have history and implement
        # TODO save while setting if you pass a save name
        # pseudocode: controls.save_params_to_history(calib_name)

    def learn_model(
        self,
        exp,
        sim,
        eval_func,
        opt_name='learn_model',
        settings={}
    ):
        # TODO allow for specific data from optimizer to be used for learning
        x0 = exp.get_parameters(self.opt_map, scaled=True)
        self.exp = exp
        self.sim = sim
        self.eval_func = eval_func
        self.opt_name = opt_name
        self.logfile_name = self.data_path + self.opt_name + '.log'
        print(f"Saving as:\n{self.logfile_name}")
        self.optim_status = {}
        self.iteration = 1

        with open(self.logfile_name, 'a') as self.logfile:
            start_time = time.time()
            start_time_str = str(f"{time.asctime(time.localtime())}\n\n")
            self.logfile.write("Starting optimization at ")
            self.logfile.write(start_time_str)
            x_best = self.Adam(
                self.goal_run_n,
                x0
            )
            self.exp.set_parameters(x_best, self.opt_map)
            end_time = time.time()
            self.logfile.write(
                f"Finished at {time.asctime(time.localtime())}\n"
            )
            self.logfile.write(
                f"Total runtime: {end_time-start_time}\n\n"
            )
            self.logfile.flush()

    def log_parameters(self, x):
        # FIXME why does log take an x parameter and doesn't use it?
        # If because callback requires it, we could print them or store them.
        self.logfile.write(json.dumps(self.optim_status))
        self.logfile.write("\n")
        self.logfile.write(f"\nStarting iteration {self.iteration}\n")
        self.iteration += 1
        self.logfile.flush()
