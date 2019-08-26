import copy
import numpy as np

import cma.evolution_strategy as cmaes


class Optimizer:

    def get_corresponding_signal_parameters(self, signals, opt_map):
        """
        Takes a dictionary of paramaters that are supposed to be optimized
        and returns the corresponding values and bounds. Writes them together
        with a tuple (key, id) to a dict and returns them

        Parameters
        ----------

        opt_map : dict
            Dictionary of parameters that are supposed to be optimized, and
            corresponding component identifier used to map
            opt_params = {
                    'T_up' : [1,2],
                    'T_down' : [1,2],
                    'freq' : [3]
                }


        Returns
        -------

        opt_params : dict
            Dictionary with values, bounds lists, plus a list with pairs of
            shape (key, id) at the corresponding pos in the list as the
            values and bounds to be able to identify the origin of the values,
            bounds in the other lists.

            Example:

            opt_params = {
                'values': [0,           0,           0,             0,             0],
                'bounds': [[0, 0],      [0, 0],      [0, 0],        [0, 0],        [0.0]],
                'origin': [('T_up', 1), ('T_up', 2), ('T_down', 1), ('T_down', 2), ('freq', 3)]
                }

        """
        opt_params = {}
        opt_params['values'] = []
        opt_params['bounds'] = []
        opt_params['origin'] = [] # array that holds tuple of (key, id) to be
                                  # identify each entry in the above lists
                                  # with it's corresponding entry

        for key in opt_map:
            for id_pair in opt_map[key]:
                signal_uuid = id_pair[0]
                for signal in signals:
                    if signal_uuid == signal.get_uuid():
                        comp_uuid = id_pair[1]
                        val = signal.get_parameter_value(key, comp_uuid)
                        bounds = signal.get_parameter_bounds(key, comp_uuid)

                        opt_params['values'].append(val)
                        opt_params['bounds'].append(bounds)
                        opt_params['origin'].append((key, id_pair))
        return opt_params


    def set_corresponding_signal_parameters(self, signals, opt_params):
        """
            sets the values in opt_params in the original signal class
        """
        for i in range(len(opt_params['origin'])):
            key = opt_params['origin'][i][0]
            id_pair = opt_params['origin'][i][1]

            signal_uuid = id_pair[0]
            comp_uuid = id_pair[1]

            for signal in signals:
                val = opt_params['values'][i]
                bounds = opt_params['bounds'][i]

                signal.set_parameter_value(key, comp_uuid, val)
                signal.set_parameter_bounds(key, comp_uuid, bounds)


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
            Array of signal parameter bounds

        Returns
        -------
        array
            Signal parameters that are compatible with bounds in physical units

        """

        values = []

        for i in range(len(x0)):
            scale = np.abs(bounds[i][0] - bounds[i][1])
            offset = bounds[i][0]

            tmp = np.arccos(np.cos((x0[i] + 1) * np.pi / 2)) / np.pi
            tmp = scale * tmp + offset
            values.append(tmp)

        return values


    def cmaes(self, opt_params, settings, eval_func, signals = None):

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

            if signals is not None:
                opt_params_clone = copy.deepcopy(opt_params)
                test_signals = []

                for sample in samples_rescaled:
                    sig_clones = copy.deepcopy(signals)
                    opt_params_clone['values'] = sample
                    self.set_corresponding_signal_parameters(sig_clones, opt_params_clone)
                    test_signals.append(sig_clones)


            if signals == None:
                eval_input = samples_rescaled
            else:
                eval_input = test_signals


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


    def optimize_signal(self, signals, opt_map, opt, settings, calib_name, eval_func):

        ####
        #
        # NOT YET TESTED
        #
        ####

        """
        Parameters
        ----------

        signal : class Signal
            Signal Class carrying all relevant information

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

        opt_params = self.get_corresponding_signal_parameters(signals, opt_map)


        if opt == 'cmaes':
            values_opt = self.cmaes(opt_params, settings, eval_func, signals)


        opt_params['values'] = values_opt


        self.set_corresponding_signal_parameters(signals, opt_params)

        for signal in signals:
            signal.save_params_to_history(calib_name)


    def optimize_gate(self,
            U0,
            gate,
            sess,
            start_name='initial',
            ol_name='open_loop'
        ):
        """
        Use minimize to optimize parameters of a given gate.

        Parameters
        ----------
        U0 : array
            Initial state represented as unitary matrix.
        gate : c3po.Gate
            Instance of the Gate class, containing control signal generation.
        start_name : str
            Name of the set of parameters to start the optimization from.
        ol_name : str
            Name of the set of parameters obtained by optimization.

        """
        params = tf.placeholder(
            tf.float64,
            shape=gate.parameters['initial'].shape
            )
        g = self.gate_err(U0, gate, params)
        jac = tf.gradients(g, params)
        res = minimize(
                lambda x: sess.run(g,
                                   feed_dict={
                                       params: gate.to_bound_phys_scale(x)
                                       }
                                   )
                ,
                gate.to_scale_one(start_name),
                jac=lambda x: sess.run(jac,
                                   feed_dict={
                                       params: gate.to_bound_phys_scale(x)
                                       }
                                   )*gate.bounds['scale']
                ,
                method='L-BFGS-B',
                options={'disp': True},
                callback=gate.print_pulse
                )
        gate.parameters[ol_name] = gate.to_bound_phys_scale(res.x)
        print('Optimal values:')
        gate.print_pulse(ol_name)

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
