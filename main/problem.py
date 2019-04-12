import utils

class Problem:
    """
    Main Class. CLass to specify the problem that we are trying to solve.

    Parameters
    ---------
    goal_func:
        maybe string/handles
    system:
        If you want to learn the model for this system.
    gates_to_optimize:
        as dict/handles or abstract objects
    initial_model:
        (optional) Can be constructed from system or given directly.
    backend: Measurement obj
        Provides different figures of merit from either experiment or
        simulation. Can contain gradients
    callback : fun
        Function to display status of the optimization process

    Attributes
    ----------
    optimal_pulses:
        Correspond to the gates_to_optimize
    model_history:
        Models improved by learning from calibration (Step III)
    gates:
        List of Gate objects
    calibration_history : dict
        Pairs of pulses and results of their respective measurement, e.g.
        fidelity.

    Methods
    -------
    get_Hamiltonian()
        model.construct_Hamiltonian() Placeholder
    optimize_pulse()
        Step I in the C3PO procedure
    calibrate_pulse()
        Step IIa in the C3PO prodecure, calibrating single gates
    calibrate_RB()
        Step IIb using randomized benchmarking
    optimize_model()
        Step III in the C3PO procedure
    """
    def __init__(self, simulation, experiment, callback=None):
        self.sim = simulation
        self.exp = experiment
        self.cbfun = callback
        self.grad_min = scp_min  # scipy optimizer as placeholder
        self.free_min = scp_min
        self.current_model = initial_model
        #TODO Solve three tasks: IBM, WMI, our simulated problem

    def optimize_pulse(gate):
        optim_opts = {''}
        x0 = gate.initial_guess.get_rescaled()
        optim_res = self.grad_min(self.sim.gate_fid(gate, self.current_model),
                x0,
                method='L-BFGS-B',
                jac=self.sim.dgate_fid(gate),
                callback=self.cbfun, optim_opts)
        gate.set_open_loop(optim_res.x)
        #TODO might have to be current

    def record_calibration(current_x, current_state):
        calib_data.append([current_x, current_state])

    def calibrate_RB(gates):
        #TODO change to multiple gates
        current_gate_name = gate.get_name()
        optim_opts = {''}
        x0 = gate.open_loop.get_rescaled()
        optim_res = self.free_min(self.exp.rb_fid(gates), x0,
                method='nelder-mead',
                callback=record_calibration,
                optim_opts)
        gate.calibrated.set(optim_res.x)

    def optimize_model_parameters():
        # uses self.calib_data

    def modify_model():
        self.current_model = changed_model

