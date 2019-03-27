import utils

#Problem Object


class System:
    """
    Abstraction of system in terms of the parts that compose it, e.g. two qubits and a cavity.
    (registers, device specs), this is what the experimentalist tells you.
    It constructs models for the system.
    
    Parameters
    ----------
    :param components:
    :param connection:
    :param phys_params:
    
    Methods
    -------
    construct_model(system_parameters, numerical_parameters)
        Construct a model for this system, to be used in numerics.
    get_parameters()
        Produces a dict of physical system parameters
    """
    
class Model:
    """
    What the theorist thinks about from the system.
    Class to store information about our system/problem/device. Different models can represent the same system.
    
    Parameters
    ---------
    physical_parameters : dict
        Represents the beta in GOAT language. Contains physical parameters as well as Hilbert space dimensions, bounds
    numerical_parameters : dict
        Hilbert space dimensions of computational and full spaces
    
    Attributes
    ----------
    H: :class:'qutip.qobj' System hamiltonian or a list of Drift and Control terms
    H_tf : empty, constructed when needed
    system_parameters : 
    numerical_parameters : 
    
    Methods
    -------
    construct_Hamiltonian(system_parameters, numerical_parameters)
        Construct a model for this system, to be used in numerics.
    get_Hamiltonian()
        Returns the Hamiltonian in a QuTip compatible way
    """
    def __init__(self, system_parameters, numerical_parameters, comp_dims):
        self.system_parameters = system_parameters
        self.numerical_parameters = numerical_parameters
        self.H = self.construct_Hamiltonian(system_parameters, numerical_parameters)
        self.projector = utils.rect_space(H[0].dims[0], comp_dims) #rect identity for computation 
        
class Problem:
    """
    Main Class. CLass to specify the problem that we are trying to solve.
    
    Parameters
    ---------
    goal_func: maybe string/handles
    gates_to_optimize: as string/handles or abstract objects
    system: If you want to learn the model for this system.
    initial_model: (optional) Can be constructed from system or given directly.
    experiment_backend: Calls the actual physical system (or simulation thereof) that you want to learn and that we will do the calibration on.
    
    Methods
    -------
    get_Hamiltonian()
        model.construct_Hamiltonian() Placeholder
    """
    

    
    
    
    
