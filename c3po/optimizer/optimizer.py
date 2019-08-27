import copy
import numpy as np
import tensorflow as tf
from scipy.optimize import minimize as minimize

import cma.evolution_strategy as cmaes


class Optimizer:

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


    def cmaes(self, opt_params, settings, eval_func, controls = None):

        ####
        #
        # NOT YET TESTED
        #
        ####

        values = opt_params['values']
        bounds = opt_params['bounds']
        if 'origin' in opt_params.keys():
            origin = opt_params['origin']



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
            samples_rescaled = [self.to_bound_phys_scale(x, bounds) for x in samples]

            if controls is not None:
                opt_params_clone = copy.deepcopy(opt_params)
                test_controls = []
                for sample in samples_rescaled:
                    sig_clones = copy.deepcopy(controls)
                    opt_params_clone['values'] = sample
                    self.set_corresponding_control_parameters(sig_clones, opt_params_clone)
                    test_controls.append(sig_clones)


            if controls == None:
                eval_input = samples_rescaled
            else:
                eval_input = test_controls


            es.tell(
                    samples,
                    eval_func(eval_input)
                    )
            es.logger.add()
            es.disp()

        res = es.result + (es.stop(), es, es.logger)
        x_opt = res[0]

        values_opt = self.to_bound_phys_scale(x_opt, bounds)

        return values_opt


    def open_loop(self, opt_params, settings, fidelity_func, controls = None):

        values, bounds = controls.get_values_bounds(opt_params)

        params = tf.placeholder(tf.float64, shape=(len(values)))

        g = fidelity_func(params, opt_params)
        jac = tf.gradients(g, params)

        x0 = self.to_scale_one(values, bounds)

        res = minimize(
                lambda x: sess.run(g,
                                   feed_dict={
                                       params: self.to_bound_phys_scale(
                                           x,
                                           bounds
                                           )
                                       }
                                   )
                ,
                x0,
                jac=lambda x: sess.run(jac,
                                   feed_dict={
                                       params: self.to_bound_phys_scale(
                                           x,
                                           bounds
                                           )
                                       }
                                   )*scale
                ,
                method='L-BFGS-B',
                options={'disp': True}
                )

        values_opt = self.to_bound_phys_scale(res.x, bounds)

        return values_opt


    def optimize_controls(self, controls, opt_map, opt, settings, calib_name, eval_func):

        ####
        #
        # NOT YET TESTED
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

        if opt == 'cmaes':
            values_opt = self.cmaes(opt_params, settings, eval_func, controls)
        elif opt == 'open_loop':
            values_opt = self.open_loop(
                opt_params,
                settings,
                eval_func,
                controls
                )

        opt_params['values'] = values_opt

        self.set_corresponding_control_parameters(opt_params)

        for control in controls:
            control.save_params_to_history(calib_name)


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
