""" Measurement object that communicates between searcher and sim/exp"""

import cma.evolution_strategy as cmaes
import numpy as np
from qutip import basis, qeye
import c3po.control.goat as goat
import tensorflow as tf
from tensorflow.python.ops.parallel_for.gradients import jacobian
from scipy.optimize import minimize as minimize
from progressbar import ProgressBar, ETA, Percentage, FileTransferSpeed, Bar


# TODO this file (measurement.py) should go in the main folder


class Backend:
    """Represents either an experiment or a simulation and contains the methods
    both need to provide.

    """

class Experiment(Backend):
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


class Simulation(Backend):
    """Short summary.

    Parameters
    ----------
    model : type
        Description of parameter `model`.
    solve_func : type
        Description of parameter `solve_func`.

    Attributes
    ----------
    solver : func
        Function that integrates the Schrödinger/Lindblad/Whatever EOM
    resolution : float
        Determined by numerical accuracy. We'll compute this from something
        like highest eigenfrequency, etc.
    model : Model
        Class that symbolically describes the model.
    """

    def __init__(self, model, solve_func, sess):
        self.model = model
        self.solver = solve_func
        self.tf_session = sess
        self.resolution = 10e9

    def update_model(self, model):
        self.model = model

    def propagation(self, U0, gate, params, do_hist=False):
        cflds, ts = gate.get_control_fields(params, self.resolution)
        h0 = self.model.tf_H0
        hks = self.model.tf_Hcs
        dt = tf.cast(ts[1], tf.complex128)
        def dU_of_t(cflds_t):
            h = h0
            for ii in range(len(hks)):
                    h += cflds_t[ii]*hks[ii]
            return tf.linalg.expm(-1j*h*dt)

        cf = tf.cast(
            tf.transpose(tf.stack(cflds)),
            tf.complex128,
            name='Control_fields'
            )

        dUs = tf.map_fn(
            dU_of_t,cf,
            name='dU_of_t'
            )

        if do_hist:
            u_t = tf.gather(dUs,0)
            history = [u_t]
            for ii in range(dUs.shape[0]-1):
                du = tf.gather(dUs, ii+1)
                u_t = tf.matmul(du,u_t)
                history.append(u_t)
            return history, ts
        else:
            U = tf.gather(dUs, 0)
            for ii in range(1, dUs.shape[0]):
                tf.matmul(tf.gather(dUs, ii), U)
            return U

    def gate_err(self, U0, gate, params):
        """
        Compute the goal function that compares the intended final state with
        actually achieved one.
        NOTE: gate_err and dgate_err should be the same function with a switch
        for gradient information. They should also not directly run the
        propagation command. Rather the propagation should be run on demand and
        the result stored, to be accessed by both fidelity functions. This
        avoids unneeded computation of dynamics.

        Parameters
        ----------
        U0 : array
            Initial state represented as unitary matrix.
        gate : c3po.Gate
            Instance of the Gate class, containing control signal generation.
        params : array
            Set of control parameters

        Returns
        -------
        array
            The final unitary.

        """
        U = self.propagation(U0, gate, params)
        U_goal = gate.goal_unitary
        return 1-tf.abs(
            tf.linalg.trace(tf.matmul(U_goal.T, U)) / U_goal.shape[1]
            )

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

    def optimize_gate(self,
            U0,
            gate,
            sess,
            start_name='initial',
            ol_name='open_loop'
        ):
        """Use GOAT to optimize parameters of a given gate.

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
        x0 = gate.to_scale_one(start_name)
        params = tf.placeholder(
            tf.float64,
            shape=gate.parameters['initial'].shape
            )
        g = self.gate_err(U0, gate, params)
        jac = tf.gradients(g, params)
        res = minimize(
                lambda x: sess.run(g,
                                   feed_dict={
                                       params: x
                                       }
                                   )
                ,
                x0,
                method='L-BFGS-B',
                jac=lambda x: sess.run(jac,
                                   feed_dict={
                                       params: x
                                       }
                                   )*gate.bounds['scale']
                ,
                options={'disp': True}
                )
        gate.parameters[ol_name] = gate.to_bound_phys_scale(res.x)
