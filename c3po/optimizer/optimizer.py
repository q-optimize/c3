import copy
import uuid
import numpy as np
import tensorflow as tf
from c3po.utils.tf_utils import tf_log10 as log10

from scipy.optimize import minimize as minimize
import cma.evolution_strategy as cmaes


class Optimizer:

    def __init__(self):
        self.sess = None
        self.store_history = False
        self.optimizer_history = []
        self.parameter_history = {}

    def set_session(self, sess):
        self.sess = sess

    def set_log_writer(self, writer):
        self.log_writer = writer

    def to_scale_one(self, values, bounds):
        """
        Returns a vector of scale 1 that plays well with optimizers.

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
        for i in range(len(values)):
            scale = np.abs(bounds[i][0] - bounds[i][1])
            offset = bounds[i][0]

            tmp = (values[i] - offset) / scale
            tmp = 2 * tmp - 1
            x0.append(tmp)

        return x0


    def to_bound_phys_scale(self, x0, bounds):
        """
        Transforms an optimizer vector back to physical scale

        Parameters
        ----------
        one : array
            Array of pulse parameters in scale 1

        bounds: array
            Array of control parameter bounds

        Returns
        -------
        array
            control parameters that are compatible with bounds in physical units

        """

        values = []

        for i in range(len(x0)):
            scale = np.abs(bounds[i][0] - bounds[i][1])
            offset = bounds[i][0]

            tmp = np.arccos(np.cos((x0[i] + 1) * np.pi / 2)) / np.pi
            tmp = scale * tmp + offset
            values.append(tmp)

        return values

    def tf_to_bound_phys_scale(self, x0, bounds):
        """
        Transforms an optimizer vector back to physical scale

        Parameters
        ----------
        one : array
            Array of pulse parameters in scale 1

        bounds: array
            Array of control parameter bounds

        Returns
        -------
        array
            control parameters that are compatible with bounds in physical units

        """
        bounds = np.array(bounds)
        scale = np.diff(bounds)
        offset = bounds.T[0]

        tmp = tf.math.acos(
            tf.math.cos((x0 + 1) * np.pi / 2)
            ) / np.pi
        values = scale.T * tmp + offset

        return tf.transpose(values)


    def fidelity_run(self, x):
        sess = self.sess
        params = self.__params
        bounds = self.bounds

        current_params = self.to_bound_phys_scale(x, bounds)

        fid = sess.run(
            self.__g,
            feed_dict={params: current_params}
            )

        if self.store_history:
            self.optimizer_history.append([current_params, fid])

        return fid


    def fidelity_run_n(self, x):
        sess = self.sess
        params = self.__params
        bounds = self.bounds

        current_params = self.to_bound_phys_scale(x, bounds)

        fid = 0

        for m in self.optimizer_history:
            fid += sess.run(
                    self.__g,
                    feed_dict={
                            params: current_params,
                            meas_result: m
                        }
                )

        if self.store_history:
            self.optimizer_history.append([current_params, fid])

        return fid


    def fidelity_gradient_run(self, x):
        sess = self.sess
        params = self.__params
        bounds = self.bounds
        scale = np.diff(bounds)

        current_params = self.to_bound_phys_scale(x,bounds)

        jac = sess.run(
                self.__jac,
                feed_dict={params: current_params}
            )

        return jac[0]*scale.T


    def fidelity_gradient_run_n(self, x):
        sess = self.sess
        params = self.__params
        bounds = self.bounds
        scale = np.diff(bounds)

        current_params = self.to_bound_phys_scale(x,bounds)

        jac = np.zeros_like(scale)

        for m in self.optimizer_history:
            jac_m = sess.run(
                    self.__jac,
                    feed_dict={
                            params: current_params,
                            meas_result: m
                        }
                )
            jac += jac_m[0]

        return jac*scale.T


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
                sample_rescaled = self.to_bound_phys_scale(sample, bounds)
                fid = self.fidelity_run(sample)
                solutions.append(fid[0][0])

            es.tell(
                    samples,
                    solutions
                    )
            es.logger.add()
            es.disp()

        res = es.result + (es.stop(), es, es.logger)
        x_opt = res[0]

        values_opt = self.to_bound_phys_scale(x_opt, bounds)

        return values_opt


    def lbfgs(self, values, bounds, goal, grad, settings={}):

        x0 = self.to_scale_one(values, bounds)

        settings['disp']=True
        res = minimize(
                goal,
                x0,
                jac=grad,
                method='L-BFGS-B',
                options=settings,
                callback=self.callback
                )

        values_opt = self.to_bound_phys_scale(res.x, bounds)

        return values_opt

    def tf_gradient_descent(self, opt_params, settings, error_func, controls):
        values, bounds = controls.get_values_bounds(opt_params)
        x0 = self.to_scale_one(values, bounds)
        params = tf.Variable(x0, dtype=tf.float64, name="params")

        opt = tf.train.GradientDescentOptimizer(
            learning_rate=0.001,
        )

        loss = log10(
            error_func(
                self.tf_to_bound_phys_scale(params, bounds),
                opt_params
                )
            )

        train = opt.minimize(loss=loss)

        init = tf.global_variables_initializer()
        self.sess.run(init)

        goal = -10
        loss_opt = 1

        while loss_opt > goal:
            self.sess.run(train)
            values_opt, loss_opt = self.sess.run([params, loss])
            #print(loss_opt, values_opt)
            print(loss_opt, self.to_bound_phys_scale(values_opt, bounds))

        return values_opt


    def optimize_controls(
        self,
        controls,
        opt_map,
        opt,
        settings,
        calib_name,
        eval_func,
        callback = None
        ):


        ####
        #
        # NOT YET THOROUGHLY TESTED
        #
        ####

        """
        Parameters
        ----------

        controls : class ControlSet
            control Class carrying all relevant information

        opt_map : dict
            Dictionary of parameters that are supposed to be optimized, and
            corresponding component identifier used to map
            opt_params = {
                    'T_up' : [1,2],
                    'T_down' : [1,2],
                    'freq' : [3]
                }

        opt : type
            Specification of the optimizer to be used, i.e. cmaes, powell, ...

        settings : dict
            Special settings for the desired optimizer
        """

        opt_params = controls.get_corresponding_control_parameters(opt_map)
        self.opt_params = opt_params

        if opt == 'cmaes':
            values, bounds = controls.get_values_bounds(opt_params)
            bounds = np.array(bounds)
            self.bounds = bounds

            params = tf.placeholder(
                tf.float64, shape=(len(values)), name="params"
                )
            self.__params = params
            self.__g = eval_func(params, opt_params)

            values_opt = self.cmaes(values, bounds, settings)

        elif opt == 'lbfgs':
            self.callback = callback
            values, bounds = controls.get_values_bounds(opt_params)
            bounds = np.array(bounds)
            self.bounds = bounds

            params = tf.placeholder(
                tf.float64, shape=(len(values)), name="params"
                )

            self.__params = params
            self.__g = eval_func(params, opt_params)
            self.__jac = tf.gradients(self.__g, params)

            values_opt = self.lbfgs(
                    values,
                    bounds,
                    self.fidelity_run,
                    self.fidelity_gradient_run,
                    settings=settings
                )

        elif opt == 'tf_grad_desc':
            values_opt = self.tf_gradient_descent(
                    opt_params,
                    settings,
                    eval_func,
                    controls
                )

        opt_params['values'] = values_opt

        controls.set_corresponding_control_parameters(opt_params)
        controls.save_params_to_history(calib_name)
        self.parameter_history[calib_name] = opt_params


    def learn_model(
        self,
        model,
        eval_func,
        settings,
        meas_results=[]
        ):

        if not meas_results == []:
            self.optimizer_history = meas_results

        values, bounds = model.get_values_bounds()
        bounds = np.array(bounds)
        self.bounds = bounds

        params = tf.placeholder(
            tf.float64, shape=(len(values)), name="params"
            )
        meas_result = tf.placeholder(tf.float64, shape=(2))

        self.__params = params
        self.__g = eval_func(params, self.opt_params, meas_result)
        self.__jac = tf.gradients(self.__g, params)

        params_opt = self.lbfgs(
                    values,
                    bounds,
                    self.fidelity_run_n,
                    self.fidelity_gradient_run_n,
                    settings=settings
                )

        model.params = np.array(params_opt)


    def sweep_bounds(self, U0, gate, n_points=101):
        spectrum = []
        range = np.linspace(0, gate.bounds['scale'], n_points)
        range += gate.bounds['offset']
        params = gate.parameters['initial']
        widgets = [
            'Sweep: ',
           Percentage(),
           ' ',
           Bar(marker='=', left='[',right=']'),
           ' ',
           ETA()
           ]
        pbar = ProgressBar(widgets=widgets, maxval=n_points)
        pbar.start()
        i=0
        for val in pbar(range):
            params[gate.opt_idxes] = val
            spectrum.append(1-self.gate_err(U0, gate, params))
            pbar.update(i)
            i+=1
        pbar.finish()
        return spectrum, range
