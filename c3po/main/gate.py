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
    # TODO make sure goal unitary is of the right dimensions
    goal: Qobj
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
            env_shape='',
            pulse={}
            ):
        self.target = target
        self.goal_unitary = goal
        # env_shape is obsolete, get rid of and deal with ETH differently
        self.env_shape = env_shape

        self.envelopes = {}
        self.opt_keys = {}
        self.keys = {}
        if pulse == {}:
            self.parameters = {}
        else:
            self.set_parameters('default', pulse)
        self.bounds = None

    def set_bounds(self, b_in):
        if self.env_shape == 'ETH':
            b = np.array(list(b_in.values()))
        else:
            self.opt_keys = self.get_keys(b_in)
            b = np.array(self.serialize_parameters(b_in, opt=True))
        self.bounds = {}
        self.bounds['scale'] = np.diff(b).T[0]
        self.bounds['offset'] = b.T[0]

    def set_parameters(self, name, guess):
        """
        An initial guess that implements this gate. The structure defines the
        parametrization of this gate.
        """
        if self.env_shape == 'ETH':
            self.parameters[name] = list(guess.values())
        else:
            self.keys = self.get_keys(guess)
            self.parameters[name] = self.serialize_parameters(guess)

    @staticmethod
    def get_keys(guess):
        keys = {}
        control_keys = sorted(guess.keys())
        for conkey in control_keys:
            control = guess[conkey]
            keys[conkey] = {}
            carrier_keys = sorted(control.keys())
            for carkey in carrier_keys:
                carrier = guess[conkey][carkey]
                keys[conkey][carkey] = {}
                pulse_keys = sorted(carrier['pulses'].keys())
                for pulkey in pulse_keys:
                    pulse = guess[conkey][carkey]['pulses'][pulkey]
                    keys[conkey][carkey][pulkey] = \
                        sorted(pulse.keys())
        return keys

    def serialize_parameters(self, p, opt=False):
        """
        Takes a nested dictionary of pulse parameters and returns a linear
        list, compatible with the parametrization of this gate. Input can
        also be the name of a stored pulse.
        """
        q = []
        if isinstance(p, str):
            p = self.parameters[p]
        if opt:
            keys = self.opt_keys
        else:
            keys = self.keys
        for conkey in sorted(keys):
            for carkey in sorted(keys[conkey]):
                q.append(p[conkey][carkey]['freq'])
                # TODO discuss adding target
                for pulkey in sorted(keys[conkey][carkey]):
                    for parkey in sorted(keys[conkey][carkey][pulkey]):
                        q.append(p[conkey][carkey]['pulses'][pulkey][parkey])
        return q

    def deserialize_parameters(self, q, opt=False):
        """
        Give a vector of parameters that conform to the parametrization for
        this gate and get the structured version back. Input can also be the
        name of a stored pulse.
        """
        p = {}
        if isinstance(q, str):
            q = self.parameters[q]
        if opt:
            keys = self.opt_keys
        else:
            keys = self.keys
        idx = 0
        for conkey in sorted(keys):
            p[conkey] = {}
            for carkey in sorted(keys[conkey]):
                p[conkey][carkey] = {}
                p[conkey][carkey]['pulses'] = {}
                p[conkey][carkey]['freq'] = q[idx]
                idx += 1
                for pulkey in sorted(keys[conkey][carkey]):
                    p[conkey][carkey]['pulses'][pulkey] = {}
                    for parkey in sorted(keys[conkey][carkey][pulkey]):
                        p[conkey][carkey]['pulses'][pulkey][parkey] = q[idx]
                        idx += 1
        return p

    def to_scale_one(self, q):
        """
        Returns a vector of scale 1 that plays well with optimizers.
        """
        if isinstance(q, str):
            q = self.parameters[q]
        y = (np.array(q) - self.bounds['offset']) / self.bounds['scale']
        return 2*y-1

    def to_bound_phys_scale(self, q):
        """
        Transforms an optimizer vector back to physical scale.
        """
        y = np.arccos(
                np.cos(
                    (np.array(q)+1)*np.pi/2
                )
            )/np.pi
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
        FED: I think this does it. However there is a problem:
        we need to make parameters and inputs of envelope match
        """
        p = self.deserialize_parameters(guess)
        keys = self.keys
        for conkey in sorted(keys):
            for carkey in sorted(keys[conkey]):
                for pulkey in sorted(keys[conkey][carkey]):
                    pars = []
                    for parkey in sorted(keys[conkey][carkey][pulkey]):
                        par = p[conkey][carkey]['pulses'][pulkey][parkey]
                        if parkey == 'type':
                            envelope = par
                        else:
                            pars.append(par)

        def Inphase(t):
            return envelope(pars) * np.cos(xy_angle)

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