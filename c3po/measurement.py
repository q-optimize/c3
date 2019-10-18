""" Measurement object that communicates between searcher and sim/exp"""

import cma.evolution_strategy as cmaes
import numpy as np
import tensorflow as tf
from scipy.optimize import minimize as minimize
from progressbar import ProgressBar, ETA, Percentage, FileTransferSpeed, Bar


# TODO this file (measurement.py) should go in the main folder

class Experiment():
    """The driver for an experiment.

    Parameters
    ----------
    eval_gate : func
        Handle for a function that takes a set of parameters and returns its
        fidelity, measured in the experiment.
    eval_seq : func
        Same as eval_gate but for a sequence of gates.

    Attributes
    ----------
    wd : str
        Working directory to store logs, plots, etc.

    """
    def __init__(self, eval_gate=None, eval_seq=None):
        """
        Initialize with eval_gate, which takes parameters for a gate and
        returns an achieved figure of merit that is to be minimized.
        """
        self.evaluate_gate = eval_gate
        self.evaluate_seq = eval_seq
        self.wd = '.'
        # TODO: Try and Handle empty function handles

    def set_working_directory(self, path):
        self.wd = path

    def calibrate_ORBIT(self, gates, opts=None, start_name='initial',
                        calib_name='calibrated', **kwargs):
        x0 = []
        ls = []
        for gate in gates:
            params = gate.rescale_and_bind(start_name)
            l_init = len(x0)
            x0.extend(params)
            l_final = len(x0)
            ls.append([l_init, l_final])
        es = cmaes.CMAEvolutionStrategy(x0,  # initial values
                                        0.5,  # initial std
                                        {'popsize': kwargs.get('popsize', 10),
                                         'tolfun': kwargs.get('tolfun', 1e-8),
                                         'maxiter': kwargs.get('maxiter', 30)}
                                        )
        iteration_number = 0
        # Main part of algorithm, like doing f_min search.
        while not es.stop():
            samples = es.ask()  # list of new solutions
            value_batch = []
            for sample in samples:
                value = []
                gate_indx = 0
                for gate in gates:
                    indeces = ls[gate_indx]
                    value.append(
                            gate.rescale_and_bind_inv(
                                sample[indeces[0]:indeces[1]]
                                )
                            )
                    gate_ind += 1
                value_batch.append(value)
            # determine RB sequences to evaluate
            sequences = sl_RB(
                    kwargs.get('n_rb_sequences', 10),
                    kwargs.get('rb_len', 20)
                    )
            # query the experiment for the survical probabilities
            results = self.evaluate_seq(sequences, value_batch)
            # tell the cmaes object the performance of each solution and update
            es.tell(samples, results)
            # log results
            es.logger.add()
            # show current evaluation status
            es.result_pretty()  # or es.disp
            # update iteration number
            iteration_number += 1
        cmaes.plot()
        return es

    def calibrate(
            self,
            gate,
            opts=None,
            start_name='initial',
            calib_name='calibrated'
            ):
        """
        Provide a gate to be calibrated with a gradient free search algorithm.
        At the moment this is CMA-ES and you can give valid opts. See pycma
        documentation for specifics. Initial sigma is set to 0.5, but you can
        give scaling for each dimension in the opts dictionary with the
        'CMA_stds' key. Further 'ftarget' sets the goal infidelity and
        'popsize' the number of samples per generation.

        Parameters
        ----------
        gate : type
            Description of parameter `gate`.
        opts : dict
            Options compatible with CMA, except for CMA_stds, which can be given in physical units.
            Example for 3 parameters:
            opts = {
                'CMA_stds' : stds,
                'ftarget' = 1e-4,
                'popsize' = 21
                }
            where stds contains the spread of the initial cloud in each
            dimension in physical units.
        start_name : str
            Name of the set of parameters to be used as initial point for the
            optimizer.
        calib_name : str
            Name for the set of parameters that the optimizer converged to.

        """
        x0 = gate.to_scale_one(start_name)
        if opts:
            if 'CMA_stds' in opts.keys():
                stds, idxes = gate.serialize_bounds(opts['CMA_stds'])
                opts['CMA_stds'] = np.array(stds)/gate.bounds['scale']

        es = cmaes.CMAEvolutionStrategy(x0, 1, opts)
        while not es.stop():
            samples = es.ask()
            samples_rescaled = [gate.to_bound_phys_scale(x) for x in samples]
            es.tell(
                    samples,
                    self.evaluate_gate(
                        gate,
                        samples_rescaled,
                        )
                    )

            es.logger.add()
            es.disp()

        res = es.result + (es.stop(), es, es.logger)
        x_opt = res[0]
        gate.parameters[calib_name] = gate.to_bound_phys_scale(x_opt)
