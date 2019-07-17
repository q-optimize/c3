import json
import numpy as np
import matplotlib.pyplot as plt


class Gate:
    """Represents a quantum gate.

    Parameters
    ----------
    target : type
        Description of parameter `target`.
    goal: array
        Unitary representation of the gate on computational subspace.
    pulse : dict
        Initial pulse parameters
    T_final : real
        Maximum time of this gate.

    Attributes
    ----------
    idxes : dict
        Contains the parametrization of this gate. This is created when
        set_parameters() is used to store a new pulse.
    opt_idxes : list
        A subset of idxes, containing the indices of parameters that are
        varied during optimization. Parameters present in the intial name
        but not in opt_idxes are frozen.
    parameters : dict
        A dictionary of linear vectors containing the parameters of
        different versions of this gate, e.g. initial guess, calibrated or
        variants.

    """

    def __init__(
            self,
            target,
            goal=None,
            pulse=None,
            T_final=100e-9
            ):

        self.T_final = T_final
        self.target = target
        self.goal_unitary = goal
        self.env_shape = None
        self.parameters = {}
        if not pulse is None:
            self.set_parameters('default', pulse)
        self.bounds = None

    def serialize_bounds(self, bounds_in):
        """Read in the bounds from a dictionary and store for rescaling.

        Parameters
        ----------
        bounds_in : dict
            A dictionary with the same structure as the pulse
            parametrization. Every dimension specified in the bounds will be
            optimized. Parameters present in the initial guess but not in the
            bounds are considered to be frozen.

        Returns
        -------
        list, list
            Linearized representation of the bounds and Indices in
            the linearized parameters that will be optimized.

        """
        opt_idxes = []
        bounds = []
        if self.env_shape == 'flat':
            for k in bounds_in:
                bounds.append(bounds_in[k])
                opt_idxes.append(self.idxes[k])
            bounds = np.array(bounds)
        else:
            for ctrl in sorted(bounds_in.keys()):
                for carr in sorted(bounds_in[ctrl].keys()):
                    for puls in sorted(
                        bounds_in[ctrl][carr]['pulses'].keys()
                        ):
                        params = (
                            bounds_in[ctrl][carr]['pulses'][puls]['params']
                        )
                        p_idx = (
                            self.idxes[ctrl][carr]['pulses'][puls]['params']
                            )
                        for prop in sorted(params.keys()):
                            opt_idxes.append(p_idx[prop])
                            bounds.append(params[prop])
        return bounds, opt_idxes


    def set_bounds(self, bounds_in):
        """
        Read in a new set of bounds for this gate. Format is the same as the
        pulse specifications but with a [min, max] at each entry.

        Parameters
        ----------
        bounds_in : dict
            The same type of dictionary as the parameter sets but with a
            [min, max] pair in each entry. The keys can be a subset of keys
            from the parameter set. Every parameter key not present here will
            be fixed during optimization.
         """
        b, self.opt_idxes = self.serialize_bounds(bounds_in)
        self.bounds = {}
        b = np.array(b)
        self.bounds['scale'] = np.diff(b).T[0]
        self.bounds['offset'] = b.T[0]

    def set_parameters(self, name, params_in):
        """
        Give an name that will define the parametrization of this gate.

        Parameters
        ----------
        name : str
            Descriptive identifier of the specified set of parameters
        params_in : dict
            Parameters in (nested) dictionary format
        """
        if self.env_shape == 'flat':
            params = []
            idxes = {}
            idx = 0
            for k in params_in:
                params.append(params_in[k])
                idxes[k] = idx
                idx += 1
            self.parameters[name] = np.array(params)
            self.idxes = idxes
        else:
            self.parameters[name] = self.serialize_parameters(params_in, True)


    def serialize_parameters(self, p, redefine=False):
        """
        Takes a nested dictionary of pulse parameters and returns a linear
        list, compatible with the parametrization of this gate. Input can
        also be the name of a stored pulse.

        Parameters
        ----------
        p : dict
            Parameters in (nested) dictionary format

        Returns
        -------
        numpy array
            Linearized parameters
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
        return np.array(q)

    def deserialize_parameters(self, q_in):
        """ Give a vector of parameters that conform to the parametrization for
        this gate and get the structured version back. Input can also be the
        name of a stored pulse.

        Parameters
        ----------
        q : array
            Numpy array containing the serialized parameters

        Returns
        -------
        type
            Description of returned object.
        """
        p = {}
        if isinstance(q_in, str):
            q_in = self.parameters[q_in]
        idxes = self.idxes
        if self.env_shape == 'flat':
            for key in idxes:
                p[key] = q_in[idxes[key]]
        else:
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
        """Returns a vector of scale 1 that plays well with optimizers.

        Parameters
        ----------
        q : array/str
            Array of parameter in physical units. Can also be the name of an array already stored in this Gate instance.

        Returns
        -------
        array
            Numpy array of pulse parameters, rescaled to values within [-1, 1]

        """
        if isinstance(q, str):
            q = self.parameters[q]
        q = np.array(q)[self.opt_idxes]
        y = (q - self.bounds['offset']) / self.bounds['scale']
        return 2*y-1

    def to_bound_phys_scale(self, x):
        """Transforms an optimizer vector back to physical scale

        Parameters
        ----------
        x : array
            Numpy array of pulse parameters in scale 1

        Returns
        -------
        array
            Pulse parameters that are compatible with bounds in physical units

        """
        y = np.arccos(np.cos((x+1)*np.pi/2))/np.pi
        q = np.array(self.parameters['initial'])
        q[self.opt_idxes] = self.bounds['scale'] * y + self.bounds['offset']
        return q

    def get_IQ(self, name, res=1e9):
        """ Construct the in-phase (I) and quadrature (Q) components of the
        control signals. These are universal to either experiment or
        simulation. In the experiment these will be routed to AWG and mixer
        electronics, while in the simulation they provide the shapes of the
        controlfields to be added to the Hamiltonian.

        Parameters
        ----------
        name : array
            Array of parameter in physical units. Can also be the name of an
            array already stored in this Gate instance.
        res : real
            Resolution of the control electronics. Will determine the number of
            time slices used to calculate waveforms.

        Returns
        -------
        dict
            Dictionary with arrays of I and Q signals for each control and
            carrier.

        """
        if isinstance(name, str):
            name = self.parameters[name]
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
                    amp = name[p_idx['amp']]
                    amp_tot_sq += amp**2
                    xy_angle = name[p_idx['xy_angle']]
                    freq_offset = name[p_idx['freq_offset']]
                    components.append(
                            amp * envelope(ts, p_idx, name)
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

        Parameters
        ----------
        name : array
            Array of parameter in physical units. Can also be the name of an
            array already stored in this Gate instanc
        res : real
            Resolution of the control electronics. Will determine the number of
            time slices used to calculate waveforms.

        Returns
        -------
        list
            List of handles for control functions.

        """
        IQ = self.get_IQ(name, res)
        """
        NICO: Federico raised the question if the xy_angle should be added
        here. After some research, this should be the correct way. The
        signal is E = I cos() + Q sin(), such that E^2 = I^2+Q^2.
        """
        cflds = []
        ts = np.linspace(0, self.T_final, int(self.T_final*res)+1)
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
        """Print out the pulse parameters in JSON format.

        Parameters
        ----------
        p : array
            Array of parameters in physical units.

        """
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

    # NICO: Do we need these? Are we strict about setting and getting? Python
    # itself doesn't seem to be.
    def get_parameters(self):
        """
        Return parameters dictionary.
        """
        return self.parameters

    def get_idxes(self):
        """
        Returns index map of parameters in the serialized vector.
        """
        return self.idxes
