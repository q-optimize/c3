import utils

#Problem Object


class System:
    """
    Abstraction of system in terms of the parts that compose it, e.g. two qubits and a cavity.
    (registers, device specs), this is what the experimentalist tells you.
    It contains model objects inside.
    :param components:
    :param connection:
    :param phys_params:
    
    """
    
    
    
    
class Model:
    """
    What the theorist thinks about from the system.
    Class to store information about our system/problem/device
    :param H: :class:'qutip.qobj' System hamiltonian or a list of Drift and Control terms
    :param comp_dims: size of the subspace that computation happens in
    """
    def __init__(self, H, system_parameters, comp_dims):
        self.H = H
        self.projector = utils.rect_space(H[0].dims[0],comp_dims) #rect identity for computation 
        
class Problem:
    """
    Main Class. CLass to specify the problem that we are trying to solve.
    :param goal_func: maybe string/handles
    :param gates_to_optimize: as string/handles or abstract objects
    """
    

    
    
    
    
