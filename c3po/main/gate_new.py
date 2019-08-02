

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
            signal=None,
            ):

        self.target = target
        self.goal_unitary = goal
        self.signal = signal

