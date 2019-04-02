import qutip as qt

class Gate:
    """
    Represents a quantum gate.

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

    def set_initial_guess(self, guess):
        self.initial_guess = guess
    
    def get_initial_guess(self):
        return self.initial_guess

    def set_name(self, name):
        self.name = name

    def get_name(self):
        return self.name

