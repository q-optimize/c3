import utils

#Problem Object

class Problem:
    """
    Class to store information about our system/problem/device
    :param H: :class:'qutip.qobj' System hamiltonian or a list of Drift and Control terms
    :param comp_dims: size of the subspace that computation happens in
    """
    def __init__(self, H, comp_dims):
        self.H = H
        self.projector = utils.rect_space(H[0].dims[0],comp_dims) #rect identity for computation 
