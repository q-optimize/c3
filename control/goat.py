import scipy.optimize._minimize as scp_min
import cma

"""The C3 protocol"""

class OC_frontend:
    """
    Hold optimal control procedures for the various steps in combined characterization and control.

    Parameters
    ----------
    problem : Problem
    fidelity : Fidelity
        Provides different figures of merit from either experiment or simulation. Can contain
        gradients.
    callback : fun
        Function to display status of the optimization process

    Attributes
    ----------
    calibration_history : dict
        Pairs of pulses and results of their respective measurement, e.g. fidelity.

    Methods
    -------
    optimize_pulse()
        Step I in the C3PO procedure
    calibrate_pulse()
        Step II in the C3PO prodecure
    optimize_model()
        Step III in the C3PO procedure
    """
    def __init__(self, problem, fidelity, callback=None):
        self.prob = problem
        self.sim_fid = fidelity.simulation
        self.cbfun = callback
        self.grad_min = scp_min  # Setup what minimizer to use for gradient based optimization here
        self.free_min = scp_min

    def optimize_pulse(gate)
        optim_opts = {''}
        x0 = gate.initial_guess.get_rescaled()
        optim_res = self.grad_min(self.sim_fid.gate(gate), x0,
                method='L-BFGS-B',
                jac=self.sim_fid.dgate(gate),
                callback=self.cbfun, optim_opts)
        gate.open_loop.set_rescaled(optim_res.x)

    def record_calibration(current_x, current_state):
        calib_entry.append([current_x, current_state])

    def calibrate_pulse(gate)
        current_gate_name = gate.get_name()
        optim_opts = {''}
        x0 = gate.open_loop.get_rescaled()
        optim_res = self.free_min(self.exp_fid.gate(gate), x0,
                method='nelder-mead',
                callback=record_calibration,
                optim_opts)
        gate.calibrated.set(optim_res.x)

