import qutip
from numpy import cos, sin

# Measurement object that communicates between searcher and sim/exp


class Backend:
    """
    Represents either an experiment or a simulation and contains the methods
    both need to provide.
    """
    def get_IQ(self, gate, name):
        """
        Construct the in-phase (I) and quadrature (Q) components of the control
        signals.
        These are universal to either experiment or simulation. In the
        experiment these will be routed to AWG and mixer electronics, while in
        the simulation they provide the shapes of the controlfields to be added
        to the Hamiltonian.
        """
        drive_parameters = gate.parameters[name]
        envelope = gate.get_envelope()
        control = drive_parameters['control1']
        carrier = control['carrier1']
        omega_d = carrier['freq']
        pulse = carrier['pulse1']
        amp = pulse['amp']
        t0 = pulse['t_up']
        t1 = pulse['t_down']
        xy_angle = pulse['xy_angle']

        def Inphase(t):
            return amp * envelope(t0, t1, t) * cos(xy_angle)

        def Quadrature(t):
            return amp * envelope(t0, t1, t) * sin(xy_angle)

        return Inphase, Quadrature, omega_d


class Experiment(Backend):
    """
    The driver for an experiment.
    """
    def __init__(self, eval_gate, eval_seq):
        self.evaluate_gate = eval_gate
        self.evaluate_seq = eval_seq
        #TODO: Try and Handle empty function handles


class Simulation(Backend):
    """
    Methods
    -------
    evolution(gate)
        constructs gate from parameters by solving equations of motion
    gate_fid(gate)
        returns findelity of gate vs gate.goal_unitary
    """
    def __init__(self, model):
        self.model = model

    def update_model(self, model):
        self.model = model

    def gate_fid(self, gate):
        U = self.evolution(gate)
        return qutip.trace_dist(U, gate.goal_unitary)

    def dgate_fid(gate):
        # horrible formula from GOAT paper
        return # gradient

    def get_control_fields(self, gate):
        """
        Returns a function handle to the control shape, constructed from drive
        parameters. For simulation we need the control fields to be added to
        the model Hamiltonian.
        """
        I, Q, omega_d = self.get_IQ(gate)
        return lambda t: I(t) * cos(omega_d * t) + Q(t) * sin(omega_d * t)
