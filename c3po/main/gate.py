import json
import numpy as np
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
            pulse={},
            T_final=100e-9
            ):
        self.T_final = T_final
        self.target = target
        self.goal_unitary = goal
        self.idxes = {}
        self.env_shape = None

        if pulse == {}:
            self.parameters = {}
        else:
            self.set_parameters('default', pulse)
        self.bounds = None

    def serialize_bounds(self, b_in):
        opt_idxes = []
        b = []
        for ctrl in sorted(b_in.keys()):
            for carr in sorted(b_in[ctrl].keys()):
                for puls in sorted(b_in[ctrl][carr]['pulses'].keys()):
                    params = b_in[ctrl][carr]['pulses'][puls]['params']
                    p_idx = self.idxes[ctrl][carr]['pulses'][puls]['params']
                    for prop in sorted(params.keys()):
                        opt_idxes.append(p_idx[prop])
                        b.append(params[prop])
        return b, opt_idxes


    def set_bounds(self, b_in):
        if self.env_shape == 'flat':
            b = np.array(list(b_in.values()))
        else:
            b, self.opt_idxes = self.serialize_bounds(b_in)
        self.bounds = {}
        b = np.array(b)
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

    def deserialize_parameters(self, q, opt=False):
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
                    params = idxes[ctrl][carr]['pulses'][puls]['params']
                    for prop in sorted(params):
                        idx = params[prop]
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
        y = np.arccos(np.cos((x+1)*np.pi/2))/np.pi
        q = np.array(self.parameters['initial'])
        q[self.opt_idxes] = self.bounds['scale'] * y + self.bounds['offset']
        return list(q)

    def get_IQ(self, guess, res=1e9):
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
        ts = np.linspace(0, self.T_final, self.T_final*res)

        for ctrl in idxes:
            ck = idxes[ctrl]
            signals[ctrl] = {}
            for carr in ck:
                Inphase = []
                Quadrature = []
                omega_d = ck[carr]['freq']
                pu = ck[carr]['pulses']
                signals[ctrl][carr] = {}
                amp_tot_sq = 0
                components = []
                for puls in pu:
                    p_idx = pu[puls]['params']
                    envelope = pu[puls]['func']
                    amp = guess[p_idx['amp']]
                    amp_tot_sq += amp**2
                    xy_angle = guess[p_idx['xy_angle']]
                    freq_offset = guess[p_idx['freq_offset']]
                    components.append(
                            amp * envelope(ts, p_idx, guess)
                            * np.exp(1j*(xy_angle+freq_offset*ts))
                            )
                norm = np.sqrt(amp_tot_sq)
                Inphase = np.real(np.sum(components, axis=0))/norm
                Quadrature = np.imag(np.sum(components, axis=0))/norm

                signals[ctrl][carr]['omega'] = omega_d
                signals[ctrl][carr]['amp'] = amp
                signals[ctrl][carr]['I'] = Inphase
                signals[ctrl][carr]['Q'] = Quadrature
        return signals

    def get_control_fields(self, name, res=1e9):
        """
        Simulation function.
        Returns a function handle to the control shape, constructed from drive
        parameters. For simulation we need the control fields to be added to
        the model Hamiltonian.
        """
        IQ = self.get_IQ(name, res)
        """
        NICO: Federico raised the question if the xy_angle should be added
        here. After some research, this should be the correct way. The
        signal is E = I cos() + Q sin(), such that E^2 = I^2+Q^2.
        """
        cflds = []
        ts = np.linspace(0, self.T_final, self.T_final*res)
        for ctrl in sorted(self.idxes):
            sig = np.zeros_like(ts)
            for carr in sorted(self.idxes[ctrl]):
                AWG_I = IQ[ctrl][carr]['I']
                AWG_Q = IQ[ctrl][carr]['Q']
                amp = IQ[ctrl][carr]['amp']
                omega_d = IQ[ctrl][carr]['omega']
                sig += amp * (
                        AWG_I * np.cos(omega_d * ts)
                        + AWG_Q * np.sin(omega_d * ts)
                         )
            cflds.append(sig)
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
        IQ = self.get_IQ(q)['control1']['carrier1']
        fig, axs = plt.subplots(2, 1)
        axs[0].plot(ts/1e-9, IQ['I'])
        axs[1].plot(ts/1e-9, IQ['Q'])
        plt.show(block=False)

    def get_parameters(self):
        return self.parameters

    def get_idxes(self):
        return self.idxes
