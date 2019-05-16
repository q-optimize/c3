""" Measurement object that communicates between searcher and sim/exp"""

import cma
from numpy import trace, zeros_like, real
from qutip import tensor, basis, qeye


class Backend:
    """
    Represents either an experiment or a simulation and contains the methods
    both need to provide.
    """


class Experiment(Backend):
    """
    The driver for an experiment.
    """
    def __init__(self, eval_gate, eval_seq=None):
        """
        Initialize with eval_gate, which takes parameters for a gate and
        returns an achieved figure of merit that is to be minimized.
        """
        self.evaluate_gate = eval_gate
        self.evaluate_seq = eval_seq
        # TODO: Try and Handle empty function handles

    def calibrate(self, gate, start_name='initial', calib_name='calibrated'):
        x0 = gate.rescale_and_bind(start_name)
        x_opt, es = cma.fmin2(
                lambda x: self.evaluate_gate(
                    gate,
                    gate.rescale_and_bind_inv(x)
                    ),
                x0,
                0.5
                )
        gate.parameters[calib_name] = gate.rescale_and_bind_inv(x_opt)
    
    def calibrate_2(self, gate, opts=None, start_name='initial', calib_name='calibrated'):
        x0 = gate.rescale_and_bind(start_name)
        if 'init_ranges' in gate.parameters.keys:
            opts['CMA_stds'] = gate.rescale_and_bind('init_ranges')

        es = cma.CMAEvolutionStrategy(x0, 1, opts)
        while not es.stop():
            samples = es.ask()
            samples_rescaled = [gate.rescale_and_bind_inv(x) for x in samples]
            es.tell(samples, self.evaluate_gate(None, samples_rescaled))
            es.logger.add()
            es.disp()
        # gate.parameters[calib_name] = gate.rescale_and_bind_inv(x_opt)


class Simulation(Backend):
    """
    Methods
    -------
    evolution(gate)
        constructs gate from parameters by solving equations of motion
    gate_fid(gate)
        returns findelity of gate vs gate.goal_unitary
    """
    def __init__(self, model, solve_func):
        self.model = model
        self.evolution = solve_func

    def update_model(self, model):
        self.model = model

    def gate_fid(self, gate):
        U = self.evolution(gate)
        U_goal = gate.goal_unitary
        g = 1-abs(trace((U_goal.dag() * U).full())) / U_goal.full().ndim
        return g

    def dgate_fid(self, gate):
        """
        Compute the gradient of the fidelity w.r.t. each parameter of the gate.
        Formally obtained by the derivative of the gate fidelity. See GOAT
        paper for details.
        """
        U = self.evolution_grad(gate)
        p = gate.parameters
        n_params = len(p) + 1
        U_goal = gate.goal_unitary
        dim = U_goal.full().ndim
        uf = tensor(basis(n_params, 0), qeye(dim)).dag() * U
        g = trace(
                (U_goal.dag() * uf).full()
            ) / dim
        ret = zeros_like(p)
        for ii in range(1, n_params):
            duf = tensor(basis(n_params, ii), qeye(dim)).dag() * U
            ret[ii-1] = -1 * real(
                g.conj() / abs(g) / dim * trace(
                    (U_goal.dag() * duf).full()
                )
            )
        return ret
