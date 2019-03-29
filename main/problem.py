import utils

#Problem Object


class System:
    """
    Abstraction of system in terms of the parts that compose it, e.g. two
    qubits and a cavity. (registers, device specs), this is what the
    experimentalist tells you. It constructs models for the system.

    Parameters
    ----------
    components: dict
        A list of physical components making up the system, e.g. qubits,
        resonators, drives
    connection: dict
        Dict of drives to each component
    phys_params: dict of dict
        Component key to a dictionary of its properties

    Methods
    -------
    construct_model(system_parameters, numerical_parameters)
        Construct a model for this system, to be used in numerics.
    get_parameters()
        Produces a dict of physical system parameters
    """

class Model:
    """
    What the theorist thinks about from the system.
    Class to store information about our system/problem/device. Different
    models can represent the same system.

    Parameters
    ---------
    physical_parameters : dict
        Represents the beta in GOAT language. Contains physical parameters as
        well as Hilbert space dimensions, bounds
    numerical_parameters : dict
        Hilbert space dimensions of computational and full spaces

    Attributes
    ----------
    H: :class:'qutip.qobj' System hamiltonian or a list of Drift and Control
        terms
    H_tf : empty, constructed when needed

    system_parameters :

    numerical_parameters :

    Methods
    -------
    construct_Hamiltonian(system_parameters, numerical_parameters)
        Construct a model for this system, to be used in numerics.
    get_Hamiltonian()
        Returns the Hamiltonian in a QuTip compatible way
    get_time_slices()
    """
    def __init__(self, system_parameters, numerical_parameters, comp_dims):
        self.system_parameters = system_parameters
        self.numerical_parameters = numerical_parameters
        self.H = self.construct_Hamiltonian(system_parameters, numerical_parameters)
        self.projector = utils.rect_space(H[0].dims[0], comp_dims) #rect identity for computation

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
    system:
        If you want to learn the model for this system.
    initial_model:
        (optional) Can be constructed from system or given directly.
    measurement : Measurement obj
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
        Step II in the C3PO prodecure
    optimize_model()
        Step III in the C3PO procedure
    """
    def __init__(self, measurement, callback=None):
        self.sim = measurement.simulation
        self.exp = measurement.experiment
        self.cbfun = callback
        self.grad_min = scp_min  # scipy optimizer as placeholder
        self.free_min = scp_min
        self.current_model = initial_model

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

    def calibrate_pulses(gates):
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

class Gate:
    """
    Represents a quantum gate.

    Parameters
    ----------
    name:
        Descriptive name, e.g. pi-gate on qubit 2
    initial_guess:
        A priori guess for parameters.
    goal_unitary: Qobj
        Unitary representation of the gate on computational subspace.

    Attributes
    ----------
    open_loop:
        Parameters after open loop OC.
    calibrated:
        Parameters after calibration on the experiment.

    Methods
    -------
    get_rescaled()
        Gives linearized parameters of a pulse scaled to order 1.
    get_params()
        Returns the dictionary of parameters in human readable form.
    get_IQ()
        Returns pulse as I and Q signals for a mixer/AWG
    """
