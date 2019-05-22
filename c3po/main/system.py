import utils

class System:
    """
    Abstraction of system in terms of the parts that compose it, e.g. two
    qubits and a cavity. (registers, device specs), this is what the
    experimentalist tells you. It constructs models for the system.

    Parameters
    ----------
    components: dict
        A list of physical components making up the system, e.g. qubits,
        resonators, drives
    connection: dict
        Dict of drives to each component
    phys_params: dict of dict
        Component key to a dictionary of its properties, not values!


    Methods
    -------
    construct_model(system_parameters, numerical_parameters)
        Construct a model for this system, to be used in numerics.
    get_parameters()
        Produces a dict of physical system parameters
    """

