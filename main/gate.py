import qutip as qt

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
    name:
        Descriptive name, e.g. pi-gate on qubit 2
    initial_guess:
        A priori guess for parameters.
    open_loop:
        Parameters after open loop OC.
    calibrated:
        Parameters after calibration on the experiment.

    Methods
    -------
    get_params()
        Returns the dictionary of parameters in human readable form.
    """

    def __init__(self, target, goal):
        self.target = target
        self.goal_unitary = goal
        self.props = ['amp', 't_up', 't_down', 'xy_angle']

    def set_initial_guess(self, guess):
        """
        An initial guess that implements this gate. The structure defines the parametrization of
        this gate.
        #TODO: this doesn't work right now. The structure needs to be stored hierarchically to
        accound for different number of pulse components in each control or carrier.
        """
        self.initial_guess = guess
        self.control_keys = sorted(guess.keys())
        for ckey in self.control_keys:
            control = guess[ckey]
            self.carrier_keys = sorted(control.keys())
            for carkey in self.carrier_keys:
                carrier = control[carkey]
                self.pulse_keys = sorted(carrier['pulses'].keys())
        
    
    def get_initial_guess(self):
        return self.initial_guess

    def get_name(self):
        return self.name

    def get_params_serialized(self, params):
        q = []
        for control_key in self.control_keys:
            control = params[control_key]
            for carrier_key in self.carrier_keys:
                carrier = control[carrier_key]
                q.append(carrier['freq'])
                for pulse_key in self.pulse_keys:
                    pulse = carrier['pulses'][pulse_key]
                    for par in self.props:
                        q.append(pulse[par])
        return q

    def get_params_deserialized(self, parvec):
        """
        Give a vector of parameters that conform to the parametrization for this gate and get the
        structured version of it back.
        """
        p_idx = 0
        params = {}
        for ckey in self.control_keys:
            control = {}
            for carkey in self.carrier_keys:
                carrier = {}
                carrier['freq'] = parvec[p_idx]
                carrier['pulses'] = {}
                p_idx += 1
                for pkey in self.pulse_keys:
                    pulse = {}
                    for par in self.props:
                        pulse[par] = parvec[p_idx]
                        p_idx += 1
                    carrier['pulses'][pkey] = pulse
                control[carkey] = carrier
            params[ckey] = control
        return params

