"""Optimizer object, where the optimal control is done."""

import random
import time
import json
import c3po
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import tensorflow_probability as tfp
from platform import python_version

from c3po.tf_utils import tf_abs
from scipy.optimize import minimize as minimize
import cma.evolution_strategy as cmaes
# from nevergrad.optimization import registry as algo_registry


class Optimizer:
    """Optimizer object, where the optimal control is done."""

    def __init__(self, cfg=None):
        # Set defaults
        self.gradients = {}
        self.noise_level = 0
        self.sampling = False
        self.batch_size = 1
        self.skip_bad_points = False  # The Millikan option, don't judge
        self.divide_by_std = False  # Goal func in terms of experiment std

        # NICO: ###############################################################
        # The default fields of this class to be stored in a config. Note: Data
        # heavy fields are excluded, as they will be transfered via logfile.
        # Maybe this should include the optimizer state in the future, to allow
        # for easier pause and repeat? A dedicated optim_state JSON might be
        # better for that.
        #######################################################################
        self.cfg_keys = [
            'noise_level', 'sampling', 'batch_size', 'skip_bad_points',
            'data_path', 'gateset_opt_map', 'opt_map',
            'opt_name', 'logfile_name'
        ]
        if cfg is not None:
            self.load_config(cfg)

    def write_config(self, filename):
        # TODO This will need to be moved to the top level script. Problem
        # Class or similar.
        cfg = {}
        cfg['title'] = "Majestic C3 config file"
        cfg['date'] = time.asctime(time.localtime())
        cfg['python_version'] = python_version()
        cfg['c3_version'] = c3po.__version__

        # Optimizer specifc code follows
        cfg['optimizer'] = {}
        cfg['optimizer']['gateset'] = self.gateset.write_config()
        cfg['optimizer']['sim'] = self.sim.write_config()
        cfg['optimizer']['exp'] = self.exp.write_config()
        for key in self.cfg_keys:
            cfg['optimizer'][key] = self.__dict__[key]

        with open(filename, "w") as cfg_file:
            json.dump(cfg, cfg_file)

    def load_config(self, filename):
        with open(filename, "r") as cfg_file:
            cfg = json.loads(cfg_file.read(1))
        for key in cfg:
            if key == 'gateset':
                self.gateset.load_config(cfg[key])
            elif key == 'sim':
                self.sim.load_config(cfg[key])
            elif key == 'exp':
                self.exp.load_config(cfg[key])
            else:
                self.__dict__[key] = cfg[key]

    def goal_run(self, current_params):
        self.gateset.set_parameters(current_params, self.opt_map, scaled=True)
        U_dict = self.sim.get_gates()
        goal = self.eval_func(U_dict)
        self.optim_status['params'] = [
            par.numpy().tolist() for par in self.exp.get_parameters(self.opt_map)
        ]
        self.optim_status['goal'] = float(goal.numpy())
        self.log_parameters()
        return goal

    def goal_run_with_grad(self, current_params):
        with tf.GradientTape() as t:
            t.watch(current_params)
            self.gateset.set_parameters(
                current_params, self.opt_map, scaled=True
            )
            U_dict = self.sim.get_gates()
            goal = self.eval_func(U_dict)

        grad = t.gradient(goal, current_params)
        gradients = grad.numpy().flatten()
        self.gradients[str(current_params.numpy())] = gradients
        self.optim_status['params'] = [
            par.numpy().tolist() for par in self.exp.get_parameters(self.opt_map)
        ]
        self.optim_status['goal'] = float(goal.numpy())
        self.optim_status['gradient'] = gradients.tolist()
        self.log_parameters()
        return goal

    def goal_run_n(self, current_params):
        learn_from = self.learn_from['seqs_grouped_by_param_set']
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
        elif self.sampling == 'ALL':
            measurements = learn_from
        else:
            raise(
                """Unspecified sampling method.\n
                Select from 'from_end'  'even', 'random' , 'from_start', 'ALL'.
                Thank you."""
            )
        batch_size = len(measurements)
        ipar = 1
        used_seqs = 0
        for m in measurements:
            gateset_params = m['params']
            self.gateset.set_parameters(
                gateset_params, self.gateset_opt_map, scaled=False
            )
            self.logfile.write(
                "\n  Parameterset {} of {}:  {}".format(
                    ipar,
                    batch_size,
                    self.gateset.get_parameters(
                        self.gateset_opt_map, to_str=True
                    )
                )
            )
            ipar += 1
            U_dict = self.sim.get_gates()
            iseq = 1
            fids = []
            sims = []
            stds = []
            for seq in m['seqs']:
                seq = seq['gate_seq']
                fid = seq['result']
                std = seq['result_std']

                if (self.skip_bad_points and fid > 0.25):
                    self.logfile.write(
                        f"\n  Skipped point with infidelity>0.25.\n"
                    )
                    iseq += 1
                    continue
                this_goal = self.eval_func(U_dict, seq)
                self.logfile.write(
                    f"\n  Sequence {iseq} of {len(m['seqs'])}:\n  {seq}\n"
                )
                iseq += 1
                self.logfile.write(
                    f"  Simulation:  {float(this_goal.numpy()):8.5f}"
                )
                self.logfile.write(
                    f"  Experiment: {fid:8.5f}"
                )
                self.logfile.write(
                    f"  Diff: {fid-float(this_goal.numpy()):8.5f}\n"
                )
                self.logfile.flush()
                used_seqs += 1

                fids.append(fid)
                sims.append(this_goal)
                stds.append(std)

            self.logfile.write(
                f"  Mean simulation fidelity: {float(np.mean(sims)):8.5f}"
            )
            self.logfile.write(
                f" std: {float(np.std(sims)):8.5f}\n"
            )
            self.logfile.write(
                f"  Mean experiment fidelity: {float(np.mean(fids)):8.5f}"
            )
            self.logfile.write(
                f" std: {float(np.std(fids)):8.5f}\n"
            )
            self.logfile.flush()

        fids = tf.constant(fids, dtype=tf.float64)
        sims = tf.concat(sims, axis=0)
        stds = tf.constant(stds, dtype=tf.float64)
        goal = self.fom(fids, sims, stds)
        self.logfile.write(
            "Finished batch with {}: {}\n".format(
                self.fom.__name__,
                float(goal.numpy())
            )
        )
        for cb_fom in self.callback_foms:
            self.logfile.write(
                "Finished batch with {}: {}\n".format(
                    cb_fom.__name__,
                    float(cb_fom(fids, sims, stds).numpy())
                )
            )
        self.logfile.flush()

        self.optim_status['params'] = [
            par.numpy().tolist() for par in self.exp.get_parameters(self.opt_map)
        ]
        self.optim_status['goal'] = float(goal.numpy())
        self.log_parameters()
        return goal

    def goal_run_n_keras(self):
        learn_from = self.learn_from
        current_params = self.keras_vars
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
        elif self.sampling == 'ALL':
            measurements = learn_from
        else:
            raise(
                """Unspecified sampling method.\n
                Select from 'from_end'  'even', 'random' , 'from_start', 'ALL'.
                Thank you."""
            )
        batch_size = len(measurements)
        ipar = 1
        used_seqs = 0
        for m in measurements:
            gateset_params = m['params']
            self.gateset.set_parameters(
                gateset_params, self.gateset_opt_map, scaled=False
            )
            self.logfile.write(
                "\n  Parameterset {} of {}:  {}".format(
                    ipar,
                    batch_size,
                    self.gateset.get_parameters(
                        self.gateset_opt_map, to_str=True
                    )
                )
            )
            ipar += 1
            U_dict = self.sim.get_gates()
            iseq = 1
            fids = []
            sims = []
            stds = []
            for seq in m['seqs']:
                seq = seq['gate_seq']
                fid = seq['result']
                std = seq['result_std']

                if (self.skip_bad_points and fid > 0.25):
                    self.logfile.write(
                        f"\n  Skipped point with infidelity>0.25.\n"
                    )
                    iseq += 1
                    continue
                this_goal = self.eval_func(U_dict, seq)
                self.logfile.write(
                    f"\n  Sequence {iseq} of {len(m[1])}:\n  {seq}\n"
                )
                iseq += 1
                self.logfile.write(
                    f"  Simulation:  {float(this_goal.numpy()):8.5f}"
                )
                self.logfile.write(
                    f"  Experiment: {fid:8.5f}"
                )
                self.logfile.write(
                    f"  Diff: {fid-float(this_goal.numpy()):8.5f}\n"
                )
                self.logfile.flush()
                used_seqs += 1

                fids.append(fid)
                sims.append(float(this_goal.numpy()))
                stds.append(std)

            self.logfile.write(
                f"  Mean simulation fidelity: {float(np.mean(sims)):8.5f}"
            )
            self.logfile.write(
                f" std: {float(np.std(sims)):8.5f}\n"
            )
            self.logfile.write(
                f"  Mean experiment fidelity: {float(np.mean(fids)):8.5f}"
            )
            self.logfile.write(
                f" std: {float(np.std(fids)):8.5f}\n"
            )
            self.logfile.flush()

        goal = self.fom(fids, sims, stds)
        self.logfile.write(
            "Finished batch with {}: {}\n".format(
                self.fom.__name__,
                float(goal.numpy())
            )
        )
        self.logfile.flush()

        self.optim_status['params'] = [
            par.numpy().tolist() for par in self.exp.get_parameters(self.opt_map)
        ]
        self.optim_status['goal'] = float(goal.numpy())
        self.log_parameters()
        return goal

    def goal_run_n_with_grad(self, current_params):
        learn_from = self.learn_from['seqs_grouped_by_param_set']
        with tf.GradientTape() as t:
            t.watch(current_params)
            self.exp.set_parameters(current_params, self.opt_map, scaled=True)

            if self.sampling == 'random':
                measurements = random.sample(learn_from, self.batch_size)
            elif self.sampling == 'even':
                n = int(len(learn_from) / self.batch_size)
                measurements = learn_from[::n]
            elif self.sampling == 'from_start':
                measurements = learn_from[:self.batch_size]
            elif self.sampling == 'from_end':
                measurements = learn_from[-self.batch_size:]
            elif self.sampling == 'ALL':
                measurements = learn_from
            else:
                raise(
                    """Unspecified sampling method.\n
                    Select from 'from_end'  'even', 'random' , 'from_start'.
                    Thank you."""
                )
            batch_size = len(measurements)
            ipar = 1
            used_seqs = 0
            for m in measurements:
                gateset_params = m['params']
                self.gateset.set_parameters(
                    gateset_params, self.gateset_opt_map, scaled=False
                )
                self.logfile.write(
                    "\n  Parameterset {} of {}:  {}".format(
                        ipar,
                        batch_size,
                        self.gateset.get_parameters(
                            self.gateset_opt_map, to_str=True
                        )
                    )
                )
                ipar += 1
                U_dict = self.sim.get_gates()
                iseq = 1
                fids = []
                sims = []
                stds = []
                for this_seq in m['seqs']:
                    seq = this_seq['gate_seq']
                    fid = this_seq['result']
                    std = this_seq['result_std']

                    if (self.skip_bad_points and fid > 0.25):
                        self.logfile.write(
                            f"\n  Skipped point with infidelity>0.25.\n"
                        )
                        iseq += 1
                        continue
                    this_goal = self.eval_func(U_dict, seq)
                    self.logfile.write(
                        f"\n  Sequence {iseq} of {len(m['seqs'])}:\n  {seq}\n"
                    )
                    iseq += 1
                    self.logfile.write(
                        f"  Simulation:  {float(this_goal.numpy()):8.5f}"
                    )
                    self.logfile.write(
                        f"  Experiment: {fid:8.5f} std: {std:8.5f}"
                    )
                    self.logfile.write(
                        f"  Diff: {fid-float(this_goal.numpy()):8.5f}\n"
                    )
                    self.logfile.flush()
                    used_seqs += 1

                    fids.append(fid)
                    sims.append(this_goal)
                    stds.append(std)

                self.sim.plot_dynamics(self.sim.ket_0, seq)

                # plt.figure()
                # signal = self.exp.generator.signal['d1']
                # plt.plot(signal['ts'], signal['values'])
                # plt.show(block=False)
                #
                # plt.figure()
                # conv_signal = self.exp.generator.devices['resp'].signal
                # plt.plot(signal['ts'], conv_signal['inphase'])
                # plt.plot(signal['ts'], conv_signal['quadrature'])
                # plt.show(block=False)

                self.logfile.write(
                    f"  Mean simulation fidelity: {float(np.mean(sims)):8.5f}"
                )
                self.logfile.write(
                    f" std: {float(np.std(sims)):8.5f}\n"
                )
                self.logfile.write(
                    f"  Mean experiment fidelity: {float(np.mean(fids)):8.5f}"
                )
                self.logfile.write(
                    f" std: {float(np.std(fids)):8.5f}\n"
                )
                self.logfile.flush()

            fids = tf.constant(fids, dtype=tf.float64)
            sims = tf.concat(sims, axis=0)
            stds = tf.constant(stds, dtype=tf.float64)
            goal = self.fom(fids, sims, stds)
            self.logfile.write(
                "Finished batch with {}: {}\n".format(
                    self.fom.__name__,
                    float(goal.numpy())
                )
            )
            for cb_fom in self.callback_foms:
                self.logfile.write(
                    "Finished batch with {}: {}\n".format(
                        cb_fom.__name__,
                        float(cb_fom(fids, sims, stds).numpy())
                    )
                )
            self.logfile.flush()

        grad = t.gradient(goal, current_params)
        gradients = grad.numpy().flatten()
        self.gradients[str(current_params.numpy())] = gradients
        self.optim_status['params'] = [
            par.numpy().tolist() for par in self.exp.get_parameters(self.opt_map)
        ]
        self.optim_status['goal'] = float(goal.numpy())
        self.optim_status['gradient'] = gradients.tolist()
        self.log_parameters()
        return goal

    def lookup_gradient(self, x):
        key = str(x)
        return self.gradients.pop(key)

    def cmaes(self, x0, goal_fun, settings={}):
        es = cmaes.CMAEvolutionStrategy(x0, 0.2, settings)
        while not es.stop():
            self.logfile.write(f"Batch {self.evaluation}\n")
            self.logfile.flush()
            samples = es.ask()
            solutions = []
            for sample in samples:
                goal = float(goal_fun(sample).numpy())
                goal = (1 + self.noise_level * np.random.randn()) * goal
                solutions.append(goal)

            self.evaluation += 1
            es.tell(
                samples,
                solutions
            )
            es.logger.add()
            es.disp()

        res = es.result + (es.stop(), es, es.logger)
        return res[0]

    # def oneplusone(self, x0, goal_fun, settings={}):
    #     optimizer = algo_registry['OnePlusOne'](instrumentation=x0.shape[0])
    #     while True:
    #         self.logfile.write(f"Batch {self.evaluation}\n")
    #         self.logfile.flush()
    #         tmp = optimizer.ask()
    #         samples = tmp.args
    #         solutions = []
    #         for sample in samples:
    #             goal = float(goal_fun(sample).numpy())
    #             solutions.append(goal)
    #             self.log_parameters(sample)
    #         self.evaluation += 1
    #         optimizer.tell(
    #             tmp,
    #             solutions
    #         )
    #
    #     recommendation = optimizer.provide_recommendation()
    #     return recommendation.args[0]

# TODO desing change? make simulator / optimizer communicate with ask and tell?
    def lbfgs(self, x0, goal, options):
        options['disp'] = True
        # Run the initial point explictly or it'll be ignored by callback
        res = minimize(
            lambda x: float(goal(x).numpy()),
            x0,
            jac=self.lookup_gradient,
            method='L-BFGS-B',
            options=options
        )
        return res.x

    def optimize_controls(
        self,
        sim,
        gateset,
        opt_map,
        opt_name,
        eval_func,
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

        algorithm : str
            Specification of the optimizer to be used, i.e. cmaes, powell, ...

        calib_name : str

        eval_func : function
            Takes the dictionary of gates and outputs a fidelity value of which
            we want to find the minimum

        settings : dict
            Special settings for the desired optimizer

        """
        # TODO Separate gateset from the simulation here.
        x0 = gateset.get_parameters(opt_map, scaled=True)
        self.init_values = x0
        self.sim = sim
        self.gateset = gateset
        self.opt_map = opt_map
        self.opt_name = opt_name
        self.eval_func = eval_func
        self.optim_status = {}
        self.evaluation = 1

        # TODO log physical values, not tf values

        self.logfile_name = self.data_path + self.opt_name + '.log'
        print(f"Saving as:\n{self.logfile_name}")
        start_time = time.time()
        with open(self.logfile_name, 'a') as self.logfile:
            start_time_str = str(f"{time.asctime(time.localtime())}\n\n")
            self.logfile.write("Starting optimization at")
            self.logfile.write(start_time_str)
            self.logfile.write(f"\n {self.opt_map}\n")
            self.logfile.flush()
            if self.algorithm == 'cmaes':
                self.cmaes(
                    x0,
                    self.goal_run,
                    settings
                )

            elif self.algorithm == 'lbfgs':
                x_best = self.lbfgs(
                    x0,
                    self.goal_run_with_grad,
                    options=settings
                )

            self.gateset.set_parameters(
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
        fom,
        callback_foms=[],
        opt_name='learn_model',
        settings={}
    ):
        # TODO allow for specific data from optimizer to be used for learning
        x0 = exp.get_parameters(self.opt_map, scaled=True)
        self.exp = exp
        self.sim = sim
        self.eval_func = eval_func
        self.fom = fom
        self.callback_foms = callback_foms
        self.opt_name = opt_name
        self.logfile_name = self.data_path + self.opt_name + '.log'
        print(f"Saving as:\n{self.logfile_name}")
        self.optim_status = {}
        self.evaluation = 0

        with open(self.logfile_name, 'a') as self.logfile:
            start_time = time.time()
            start_time_str = str(f"{time.asctime(time.localtime())}\n\n")
            self.logfile.write("Starting optimization at ")
            self.logfile.write(start_time_str)
            self.logfile.write("Optimization parameters:\n\n")
            self.logfile.write(json.dumps(self.opt_map))
            # TODO put optmizer specific code here
            if self.algorithm == 'cmaes':
                x_best = self.cmaes(
                    x0,
                    lambda x: self.goal_run_n(tf.constant(x)),
                    settings
                )

            elif self.algorithm == 'lbfgs':
                x_best = self.lbfgs(
                    x0,
                    lambda x: self.goal_run_n_with_grad(tf.constant(x)),
                    options=settings
                )

            elif self.algorithm == 'oneplusone':
                x_best = self.oneplusone(
                    tf.constant(x0),
                    self.goal_run_n,
                    options=settings
                )

            elif self.algorithm == 'keras-SDG':
                vars = tf.Variable(x0)
                self.keras_vars = vars
                optimizer = tf.keras.optimizers.SGD(learning_rate=0.1)
                optimizer.minimize(self.goal_run_n_keras, var_list=[vars])
                x_best = vars.numpy()

            elif self.algorithm == 'keras-Adam':
                vars = tf.Variable(x0)
                optimizer = tf.keras.optimizers.Adam(learning_rate=0.1)
                optimizer.minimize(self.goal_run_n(vars), var_list=[vars])
                x_best = vars.numpy()

            else:
                raise Exception(
                    "I don't know the selected optimization algorithm."
                )
            self.exp.set_parameters(
                x_best, self.opt_map, scaled=True
            )
            end_time = time.time()
            self.logfile.write(
                f"Finished at {time.asctime(time.localtime())}\n"
            )
            self.logfile.write(
                f"Total runtime: {end_time-start_time}\n\n"
            )
            self.logfile.flush()

    def model_1d_sweep(
        self,
        exp,
        sim,
        eval_func,
        opt_name='model_sweep',
        num=50,
    ):
        from progressbar import ProgressBar, Percentage, Bar, ETA
        import matplotlib.pyplot as plt

        self.exp = exp
        self.sim = sim
        self.eval_func = eval_func
        self.opt_name = opt_name
        self.logfile_name = self.data_path + self.opt_name + '.log'
        print(f"Saving as:\n{self.logfile_name}")
        self.optim_status = {}
        self.evaluation = 1

        with open(self.logfile_name, 'a') as self.logfile:
            X = np.linspace(-1, 1, num)
            rms = []
            widgets = [
                'Sweep: ',
                Percentage(),
                ' ',
                Bar(marker='=', left='[', right=']'),
                ' ',
                ETA()
            ]
            pbar = ProgressBar(widgets=widgets, maxval=X.shape[0])
            pbar.start()
            ii = 0
            for val in pbar(range(X.shape[0])):
                rms.append(self.goal_run_n([X[ii]]))
                pbar.update(ii)
                ii += 1
            pbar.finish()
            plt.figure()
            plt.plot(X, rms, 'x')
            plt.show()

    def log_parameters(self):
        self.logfile.write(json.dumps(self.optim_status))
        self.logfile.write("\n")
        self.logfile.write(f"\nStarting evaluation {self.evaluation}\n")
        self.evaluation += 1
        self.logfile.flush()
