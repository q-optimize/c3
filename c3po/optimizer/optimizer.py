import numpy as np
import cma.evolution_strategy as cmaes


class Optimizer:

    #####
    #
    # TODO: split up this function into a pair of get and set functions
    #       for accessing single parameter values/bounds of a Signal obj
    #
    #####

    def get_corresponding_signal_parameters(self, signal, opt_params):
        opt_params['values'] = []
        opt_params['bounds'] = []
        opt_params['resolv'] = []

        for key in opt_params:
            for comp_id in opt_params[key]:
                for comp in signal.comps:
                    if comp_id == comp.get_id():
                        opt_params['values'].append(comp.params[key])
                        opt_params['bounds'].append(comp.bounds[key])
                        opt_params['resolv'].append((key, comp_id))


    def set_corresponding_signal_parameters(self, signal, opt_params):

        for i in range(len(opt_params['resolv']) - 1):
            key = opt_params['resolv'][i][0]
            comp_id = opt_params['resolv'][i][1]

            for comp in signal.comps:
                if comp_id == comp.get_id():
                    comp.params[key] = opt_params['values'][i]
                    comp.bounds[key] = opt_params['bounds'][i]


    def to_scale_one(self, values, bounds):
        """Returns a vector of scale 1 that plays well with optimizers.

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
        """Transforms an optimizer vector back to physical scale

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

        for i in range(len(one)):
            scale = np.abs(bounds[i][0] - bounds[i][1])
            offset = bounds[i][0]

            tmp = np.arccos(np.cos((x0[i] + 1) * np.pi / 2)) / np.pi
            tmp = scale * tmp + offset
            values.append(tmp)

        return values


    def cmaes_opt(self, signal, x0, bounds, settings, eval_func):

        ####
        #
        # YET TO BE TESTED
        #
        ####

        if settings:
            if 'CMA_stds' in settings.keys():
                scale_bounds = []

                for i in range(len(bounds)):
                    scale = np.abs(bounds[i][0] - bounds[i][1])
                    scale_bounds.append(settings['CMA_stds'][i] / scale)

                settings['CMA_stds'] = scale_bounds


        es = cmaes.CMAEvolutionStrategy(x0, 1, settings)

        while not es.stop():
            samples = es.ask()
            samples_rescaled = [self.to_bound_phys_scale(x, bounds) for x in samples]
            es.tell(
                    samples,
                    eval_func(
                        signal,
                        samples_rescaled,
                        )
                    )
            es.logger.add()
            es.disp()

        res = es.result + (es.stop(), es, es.logger)
        x_opt = res[0]

        return x_opt




    def optimize_signal(self, signal, opt_params, opt, settings, calib_name, eval_func):

        ####
        #
        # YET TO BE TESTED
        #
        ####

        """
        Parameters
        ----------

        signal : class Signal
            Signal Class carrying all relevant information

        opt_params : dict
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

        self.get_corresponding_signal_parameters(signal, opt_params)

        values = opt_params['values']
        bounds = opt_params['bounds']


        x0 = self.to_scale_one(values, bounds)

        if opt == 'cmaes':
            x_opt = self.cmaes_opt(x0, bounds, settings, eval_func)

        values_opt = self.to_bound_phys_scale(x_opt, bounds)

        opt_params['values'] = values_opt

        self.set_corresponding_signal_parameters(signal, opt_params)

        signal.save_params_to_history(calib_name)







