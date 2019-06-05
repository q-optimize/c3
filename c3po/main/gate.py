import json
import numpy as np
from c3po.utils.envelopes import flattop, gaussian, gaussian_der
from c3po.utils.helpers import sum_lambdas
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
    idxes:
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
            pulse={},
            T_final=100e-9
            ):
        self.T_final = T_final
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
        elif env_shape == 'flat':
            props = ['amplitude', 'length', 'alpha']
            env_func = None
        self.props = props
        self.envelope = env_func

        self.idxes = {}
        if pulse == {}:
            self.parameters = {}
        else:
            self.set_parameters('default', pulse)

        self.bounds = None

    def set_bounds(self, b_in):
        if self.env_shape == 'flat':
            b = np.array(list(b_in.values()))
        else:
            opt_idxes = []
            idxes = self.idxes
            b = []
            for ctrl in sorted(b_in.keys()):
                for carr in sorted(b_in[ctrl].keys()):
                    for puls in sorted(b_in[ctrl][carr]['pulses'].keys()):
                        for prop in sorted(
                                b_in[ctrl][carr]['pulses'][puls]['params'].keys()
                                ):
                            opt_idxes.append(
                                idxes[ctrl][carr]['pulses'][puls]['params'][prop]
                                )
                            b.append(
                                b_in[ctrl][carr]['pulses'][puls]['params'][prop]
                                )
        self.bounds = {}
        b = np.array(b)
        self.opt_idxes = opt_idxes
        self.bounds['scale'] = np.diff(b).T[0]
        self.bounds['offset'] = b.T[0]

    def set_parameters(self, name, guess):
        """
        An initial guess that implements this gate. The structure defines the
        parametrization of this gate.
        """
        if self.env_shape == 'flat':
            self.parameters[name] = list(guess.values())
        else:
            self.parameters[name] = self.serialize_parameters(guess, True)

    def serialize_parameters(self, p, redefine=False):
        """
        Takes a nested dictionary of pulse parameters and returns a linear
        list, compatible with the parametrization of this gate. Input can
        also be the name of a stored pulse.
        """
        q = []
        idx = 0
        idxes = {}
        for ctrl in sorted(p.keys()):
            idxes[ctrl] = {}
            for carr in sorted(p[ctrl].keys()):
                idxes[ctrl][carr] = {}
                idxes[ctrl][carr]['freq'] = p[ctrl][carr]['freq']
                idxes[ctrl][carr]['pulses'] = {}
                # TODO discuss adding target
                for puls in sorted(p[ctrl][carr]['pulses'].keys()):
                    idxes[ctrl][carr]['pulses'][puls] = {}
                    idxes[ctrl][carr]['pulses'][puls]['func']\
                        = p[ctrl][carr]['pulses'][puls]['func']
                    idxes[ctrl][carr]['pulses'][puls]['params'] = {}
                    for prop in sorted(
                            p[ctrl][carr]['pulses'][puls]['params'].keys()
                            ):
                        idxes[ctrl][carr]['pulses'][puls]['params'][prop] = idx
                        q.append(p[ctrl][carr]['pulses'][puls]['params'][prop])
                        idx += 1
        if redefine:
            self.idxes = idxes
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
        idxes = self.idxes
        for ctrl in sorted(idxes):
            p[ctrl] = {}
            for carr in sorted(idxes[ctrl]):
                p[ctrl][carr] = {}
                p[ctrl][carr]['pulses'] = {}
                p[ctrl][carr]['freq'] = idxes[ctrl][carr]['freq']
                for puls in sorted(idxes[ctrl][carr]['pulses']):
                    p[ctrl][carr]['pulses'][puls] = {
                            'params': {}
                            }
                    for prop in sorted(idxes[ctrl][carr]['pulses'][puls]['params']):
                        idx = idxes[ctrl][carr]['pulses'][puls]['params'][prop]
                        p[ctrl][carr]['pulses'][puls]['params'][prop] = q[idx]
        return p

    def to_scale_one(self, q):
        """
        Returns a vector of scale 1 that plays well with optimizers.
        """
        if isinstance(q, str):
            q = self.parameters[q]
        q = np.array(q)[self.opt_idxes]
        y = (q - self.bounds['offset']) / self.bounds['scale']
        return 2*y-1

    def to_bound_phys_scale(self, x):
        """
        Transforms an optimizer vector back to physical scale.
        """
        y = np.arccos(np.cos(x+1)*np.pi/2)/np.pi
        q = np.array(self.parameters['initial'])
        q[self.opt_idxes] = self.bounds['scale'] * y + self.bounds['offset']
        return list(q)

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
        idxes = self.idxes
        signals = {}

        for ctrl in idxes:
            ck = idxes[ctrl]
            signals[ctrl] = {}
            for carr in ck:
                Inphase = []
                Quadrature = []
                omega_d = guess[ck[carr]['freq']]
                pu = ck[carr]['pulses']
                comp_amps = []
                components = []

                for puls in pu:
                    amp = guess[pu[puls]['amp']]
                    t0 = guess[pu[puls]['t_up']]
                    t1 = guess[pu[puls]['t_down']]
                    xy_angle = guess[pu[puls]['xy_angle']]
                    comp_amps.append(amp)
                    components.append(
                        lambda t:
                            amp * self.envelope(t, t0, t1)
                            * np.exp(1j*xy_angle)
                        )

                def Inphase(t):
                    return np.real(sum_lambdas(t, components))/max(comp_amps)

                def Quadrature(t):
                    return np.imag(sum_lambdas(t, components))/max(comp_amps)

                signals[ctrl][carr]['omega'] = omega_d
                signals[ctrl][carr]['amp'] = max(comp_amps)
                signals[ctrl][carr]['I'] = Inphase
                signals[ctrl][carr]['Q'] = Quadrature
        return signals

    def get_control_fields(self, name):
        """
        Simulation function.
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
        cflds = []
        for ckey in sorted(self.keys):
            cflds.append(
                lambda t:
                    amp * (
                        mixer_I(t) * tf.cos(omega_d * t)
                        + mixer_Q(t) * tf.sin(omega_d * t)
                         )
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
        ts = np.linspace(0, self.T_final, self.T_final*1e9)
        plt.rcParams['figure.dpi'] = 100
        IQ = self.get_IQ(q)
        fig, axs = plt.subplots(2, 1)
        axs[0].plot(ts/1e-9, list(map(IQ['I'], ts)))
        axs[1].plot(ts/1e-9, list(map(IQ['Q'], ts)))
        plt.show(block=False)

    def get_parameters(self):
        return self.parameters

    def get_idxes(self):
        return self.idxes
