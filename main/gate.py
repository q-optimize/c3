import json
from numpy import cos, sin


class Gate:
    """
    Represents a quantum gate with a fixed parametrization and envelope shape.

    Parameters
    ----------
    target: Component
        Model component(s) to act upon
    goal_unitary: Qobj
        Unitary representation of the gate on computational subspace.

    Attributes
    ----------
    keys:
        Contains the parametrization of this gate. This is created when
        set_parameters() is used to store a new pulse.
    parameters:
        A dictionary of linear vectors containing the parameters of different
        versions of this gate, e.g. initial guess, calibrated or variants.
    props:
        Dictionary of properties of the pulse components, specific to the
        envelope function. Note: fixed for now. Will later be initialized.
    envelope:
        Function handle from our extensive library of shapes, a flattop mostly
    """

    def __init__(self, target, goal, env_func):
        self.target = target
        self.goal_unitary = goal
        self.props = ['amp', 't_up', 't_down', 'xy_angle']
        self.envelope = env_func
        self.keys = {}
        self.parameters = {}

    def set_parameters(self, name, guess):
        """
        An initial guess that implements this gate. The structure defines the
        parametrization of this gate.
        """
        control_keys = sorted(guess.keys())
        for ckey in control_keys:
            control = guess[ckey]
            self.keys[ckey] = {}
            carrier_keys = sorted(control.keys())
            for carkey in carrier_keys:
                carrier = guess[ckey][carkey]
                self.keys[ckey][carkey] = sorted(carrier['pulses'].keys())
        self.parameters[name] = self.serialize_parameters(guess)

    def serialize_parameters(self, p):
        """
        Takes a nested dictionary of pulse parameters and returns a linear
        vector, compatible with the parametrization of this gate. Input can
        also be the name of a stored pulse.
        """
        q = []
        if isinstance(p, str):
            p = self.parameters[p]
        keys = self.keys
        for ckey in sorted(keys):
            for carkey in sorted(keys[ckey]):
                q.append(p[ckey][carkey]['freq'])
                for pkey in sorted(keys[ckey][carkey]):
                    for prop in self.props:
                        q.append(p[ckey][carkey]['pulses'][pkey][prop])
        return q

    def deserialize_parameters(self, q):
        """
        Give a vector of parameters that conform to the parametrization for
        this gate and get the structured version back. Input can also be the
        name of a stored pulse.
        """
        p = {}
        if isinstance(q, str):
            q = self.parameters[q]
        keys = self.keys
        idx = 0
        for ckey in sorted(keys):
            p[ckey] = {}
            for carkey in sorted(keys[ckey]):
                p[ckey][carkey] = {}
                p[ckey][carkey]['pulses'] = {}
                p[ckey][carkey]['freq'] = q[idx]
                idx += 1
                for pkey in sorted(keys[ckey][carkey]):
                    p[ckey][carkey]['pulses'][pkey] = {}
                    for prop in self.props:
                        p[ckey][carkey]['pulses'][pkey][prop] = q[idx]
                        idx += 1
        return p

    def get_IQ(self, name):
        """
        Construct the in-phase (I) and quadrature (Q) components of the control
        signals.
        These are universal to either experiment or simulation. In the
        experiment these will be routed to AWG and mixer electronics, while in
        the simulation they provide the shapes of the controlfields to be added
        to the Hamiltonian.
        """
        drive_parameters = self.parameters[name]
        envelope = self.envelope
        omega_d = drive_parameters[0]
        amp = drive_parameters[1]
        t0 = drive_parameters[2]
        t1 = drive_parameters[3]
        xy_angle = drive_parameters[4]

        def Inphase(t):
            return amp * envelope(t, t0, t1) * cos(xy_angle)

        def Quadrature(t):
            return amp * envelope(t, t0, t1) * sin(xy_angle)

        return Inphase, Quadrature, omega_d

    def get_control_fields(self, gate):
        """
        Returns a function handle to the control shape, constructed from drive
        parameters. For simulation we need the control fields to be added to
        the model Hamiltonian.
        """
        I, Q, omega_d = self.get_IQ(gate)
        return lambda t: I(t) * cos(omega_d * t) + Q(t) * sin(omega_d * t)

    def print(self, p):
        print(json.dumps(
            self.deserialize_parameters(p),
            indent=4,
            sort_keys=True))
    def get_parameters(self):
        return self.parameters

    def get_keys(self):
        return self.keys

