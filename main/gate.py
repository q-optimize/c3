import json
import numpy as np
from c3po.utils.envelopes import flattop, gaussian, gaussian_der
import matplotlib.pyplot as plt


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

    def __init__(
            self,
            target,
            goal,
            env_shape='flattop',
            pulse={}):
        self.target = target
        self.goal_unitary = goal
        self.env_shape = env_shape
        if env_shape == 'gaussian':
            # TODO figure out parallel imports
            env_func = gaussian
            env_der = gaussian_der
            self.env_der = env_der
            props = ['amp', 'T', 'sigma', 'xy_angle']
        elif env_shape == 'flattop':
            env_func = flattop
            props = ['amp', 't_up', 't_down', 'xy_angle']
        elif env_shape == 'flattop_risefall':
            env_func = flattop
            props = ['amp', 't_up', 't_down', 'xy_angle', 'T', 'risefall']
        elif env_shape == 'DRAG':
            env_func = gaussian
            env_der = gaussian_der
            self.env_der = env_der
            props = ['amp', 'T', 'sigma', 'xy_angle', 'drag']
        self.props = props
        self.envelope = env_func

        self.keys = {}
        if pulse == {}:
            self.parameters = {}
        else:
            self.set_parameters('default', pulse)

        self.bounds = None

    def set_bounds(self, b_in):
        b = np.array(self.serialize_parameters(b_in))
        self.bounds = {}
        self.bounds['scale'] = np.diff(b).T[0]
        self.bounds['offset'] = b.T[0]

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
        list, compatible with the parametrization of this gate. Input can
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

    def rescale_and_bind(self, q):
        """
        Returns a vector of scale 1 that plays well with optimizers.
        """
        if isinstance(q, str):
            q = self.parameters[q]
        x = (np.array(q) - self.bounds['offset']) / self.bounds['scale']
        return np.arccos(2 * x - 1)

    def rescale_and_bind_inv(self, x):
        """
        Transforms an optimizer vector back to physical scale.
        """
        y = (np.cos(np.abs(x))+1)/2
        return self.bounds['scale'] * y + self.bounds['offset']

    def get_IQ(self, guess):
        """
        Construct the in-phase (I) and quadrature (Q) components of the control
        signals.
        These are universal to either experiment or simulation. In the
        experiment these will be routed to AWG and mixer electronics, while in
        the simulation they provide the shapes of the controlfields to be added
        to the Hamiltonian.
        """
        if isinstance(guess, str):
            guess = self.parameters[guess]
        """
        NICO: Paramtrization here is fixed for testing and will have to be
        extended to more general.
        """
        omega_d = guess[0]
        amp = guess[1]
        t0 = guess[2]
        t1 = guess[3]
        xy_angle = guess[4]
        # TODO: atm it works for both gaussian and flattop, but only by chance

        def Inphase(t):
            return self.envelope(t, t0, t1) * np.cos(xy_angle)

        def Quadrature(t):
            envelope = self.envelope
            if self.env_shape == 'DRAG':
                drag = guess[5]
                envelope = drag * self.env_der
            return envelope(t, t0, t1) * np.sin(xy_angle)

        return {
                'I': Inphase,
                'Q': Quadrature,
                'carrier_amp': amp,
                'omegas': [omega_d]
                }

    def get_control_fields(self, name):
        """
        Returns a function handle to the control shape, constructed from drive
        parameters. For simulation we need the control fields to be added to
        the model Hamiltonian.
        """
        p_IQ = self.get_IQ(name)
        mixer_I = p_IQ['I']
        mixer_Q = p_IQ['Q']
        omega_d = p_IQ['omegas'][0]
        amp = p_IQ['carrier_amp']
        """
        NICO: Federico raised the question if the xy_angle should be added
        here. After some research, this should be the correct way. The
        signal is E = I cos() + Q sin(), such that E^2 = I^2+Q^2.
        """
        return lambda t:\
            amp * (mixer_I(t) * np.cos(omega_d * t)
                   + mixer_Q(t) * np.sin(omega_d * t))

    def print_pulse(self, p):
        print(
                json.dumps(
                    self.deserialize_parameters(p),
                    indent=4,
                    sort_keys=True
                    )
            )

    def plot_control_fields(self, q='initial', axs=None):
        """ Plotting control functions """
        ts = np.linspace(0, 100e-9, 100)
        plt.rcParams['figure.dpi'] = 100
        IQ = self.get_IQ(q)
        fig, axs = plt.subplots(2, 1)
        axs[0].plot(ts/1e-9, list(map(IQ['I'], ts)))
        axs[1].plot(ts/1e-9, list(map(IQ['Q'], ts)))
        plt.show(block=False)
