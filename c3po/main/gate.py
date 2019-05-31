import json
import numpy as np
import tensorflow as tf
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
            tf_sess,
            env_shape='flattop',
            pulse={},
            ):
        self.target = target
        self.goal_unitary = goal
        self.env_shape = env_shape
        if env_shape == 'gaussian':
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
        elif env_shape == 'ETH':
            props = ['amplitude', 'length', 'alpha']
            env_func = None
        self.props = props
        self.envelope = env_func

        self.keys = {}
        if pulse == {}:
            self.parameters = {}
        else:
            self.set_parameters('default', pulse)

        self.bounds = None

        self.tf_sess = tf_sess

    def set_bounds(self, b_in):
        if self.env_shape == 'ETH':
            b = tf.constant(list(b_in.values()))
        else:
            b = tf.constant(
                    self.serialize_parameters(b_in),
                    dtype=tf.float64
                    )
        self.bounds = {}
        self.bounds['scale'] = b[:, 1]-b[:, 0]
        self.bounds['offset'] = b[:, 0]

    def set_parameters(self, name, guess):
        """
        An initial guess that implements this gate. The structure defines the
        parametrization of this gate.
        """
        if self.env_shape == 'ETH':
            self.parameters[name] = list(guess.values())
        else:
            control_keys = sorted(guess.keys())
            for ckey in control_keys:
                control = guess[ckey]
                self.keys[ckey] = {}
                carrier_keys = sorted(control.keys())
                for carkey in carrier_keys:
                    carrier = guess[ckey][carkey]
                    self.keys[ckey][carkey] = sorted(carrier['pulses'].keys())
            self.parameters[name] = tf.constant(
                    self.serialize_parameters(guess),
                    dtype=tf.float64
                    )

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
                # TODO discuss adding target
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
        q = self.tf_sess.run(q)
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

    def to_scale_one(self, q):
        """
        Returns a vector of scale 1 that plays well with optimizers. Input type
        is Tensor or a String that identifies a stored Tensor, i.e 'initial'.
        """
        if isinstance(q, str):
            q = self.parameters[q]
        y = (q - self.bounds['offset']) / self.bounds['scale']
        return 2*y-1

    def to_bound_phys_scale(self, x):
        """
        Transforms an optimizer vector back to physical scale.
        """
        y = tf.acos(
                tf.cos(
                    (x+1)*np.pi/2
                )
            )/np.pi
        return self.bounds['scale'] * y + self.bounds['offset']

    def get_IQ(self, pulses):
        """
        Construct the in-phase (I) and quadrature (Q) components of the control
        signals.
        These are universal to either experiment or simulation. In the
        experiment these will be routed to AWG and mixer electronics, while in
        the simulation they provide the shapes of the controlfields to be added
        to the Hamiltonian.
        """
        """
        NICO: Paramtrization here is fixed for testing and will have to be
        extended to more general.
        """
        # TODO: atm it works for both gaussian and flattop, but only by chance
        Inphase = []
        Quadrature = []

        for p_name in pulses:
            pulse = pulses[p_name]
            t0 = pulse['t_up']
            t1 = pulse['t_down']
            xy_angle = pulse['xy_angle']
            amp = pulse['amp']

            Inphase.append(
                    lambda t: amp * self.envelope(t, t0, t1) * tf.cos(xy_angle)
                    )
            Quadrature.append(
                    lambda t: amp * self.envelope(t, t0, t1) * tf.sin(xy_angle)
                    )

        return Inphase, Quadrature

    def get_control_fields(self, guess):
        """
        Returns a function handle to the control shape, constructed from drive
        parameters. For simulation we need the control fields to be added to
        the model Hamiltonian.
        """
        """
        NICO: Federico raised the question if the xy_angle should be added
        here. After some research, this should be the correct way. The
        signal is E = I cos() + Q sin(), such that E^2 = I^2+Q^2.
        """
        if isinstance(guess, str):
            guess = self.parameters[guess]
        p = self.deserialize_parameters(guess)
        cflds = []
        for ckey in sorted(self.keys):
            for carkey in sorted(self.keys[ckey]):
                pulses = p[ckey][carkey]['pulses']
                mixer_Is, mixer_Qs = self.get_IQ(pulses)

                def mixer_I(t):
                    return sum(f(t) for f in mixer_Is)

                def mixer_Q(t):
                    return sum(f(t) for f in mixer_Qs)

                omega_d = p[ckey][carkey]['freq']
                cflds.append(
                    lambda t:
                        mixer_I(t) * tf.cos(omega_d * t)
                        + mixer_Q(t) * tf.sin(omega_d * t)
                    )
        return cflds

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
