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
    def __init__(self, send_pulse, recv_answer):
        self.send_pulse = send_pulse
        self.recv_answer = recv_answer


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

    def evolution(self, gate):
        """
        Sets up the time evolution of running a given gate on the model. The
        gate needs to contain the envelope function handle and drive
        parameters.
        """
        drv_par = gate.get_params()
        env = gate.get_envelope()
        h = self.model.get_Hamiltonian(self.get_control_fields(drv_par, env))
        ts = self.model.get_time_slices()
        params = gate.get_params_linear()
        return utils.evolution(h, ts, params)

    def gate_fid(self, gate):
        U = self.evolution(gate)
        return qutip.trace_dist(U, gate.goal_unitary)

    def get_control_fields(self, drive_parameters, envelope):
        """
        Returns a function handle to the control shape, constructed from drive
        parameters. For simulation we need the control fields to be added to
        the model Hamiltonian.
        """
        I, Q, omega_d = self.get_IQ(drive_parameters, envelope)
        return lambda t: I(t) * cos(omega_d * t) + Q(t) * sin(omega_d * t)
