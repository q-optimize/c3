import utils

# Measurement object that communicates between searcher and sim/exp

class Backend:


class Experiment(Backend):

class Simulation(Backend):
    """
    Methods
    -------
    evolution(gate)
        constructs gate from parameters by solving equations of motion
    gate_fid(gate)
        returns findelity of gate vs gate.goal_unitary
    """
    def evolution(gate, model):
        h = model.get_Hamiltonian()
        ts = model.get_time_slices()
        params = gate.get_params()
        return utils.evolution(h, ts, params)

    def gate_fid(gate, model):
        U = self.evolution(gate, model)
        return qutip.trace_dist(U,gate.goal_unitary)

class Measurement:
    """
    simulation_backend:
        Open loop. Provides measurement results from simulated dynamics.
    experiment_backend:
        Closed loop. Calls the actual physical system (or simulation thereof)
        that you want to learn and that we will do the calibration on.
    """

