""" Measurement object that communicates between searcher and sim/exp"""

import cma
from numpy import trace, zeros_like, real
from qutip import tensor, basis, qeye

# TODO this file (measurement.py) should go in the main folder
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

    def calibrate_ORBIT(gates):
        return calibrate_ORBIT

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

        Example for 3 parameters:
        ops = {
            'CMA_stds' : [1, 2, 0.5],
            'ftarget' = 1e-4,
            'popsize' = 20,
            }
        """
        x0 = gate.to_scale_one(start_name)
        es = cma.CMAEvolutionStrategy(x0, 0.5, opts)
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
        # TODO shouldn't this be squared
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
