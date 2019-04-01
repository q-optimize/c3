class Model:
    """
    What the theorist thinks about from the system.
    Class to store information about our system/problem/device. Different
    models can represent the same system.

    Parameters
    ---------
    physical_parameters : dict
        Represents the beta in GOAT language. Contains physical parameters as
        well as Hilbert space dimensions, bounds
    hilbert_space : dict
        Hilbert space dimensions of computational and full spaces

    Attributes
    ----------
    H: :class:'qutip.qobj' System hamiltonian or a list of Drift and Control
        terms
    H_tf : empty, constructed when needed

    component_parameters :

    hilbert_space :

    Methods
    -------
    construct_Hamiltonian(component_parameters, hilbert_space)
        Construct a model for this system, to be used in numerics.
    get_Hamiltonian()
        Returns the Hamiltonian in a QuTip compatible way
    get_time_slices()
    """
    def __init__(self, component_parameters, coupling_strength, hilbert_space, comp_dims):
        global hbar

        self.component_parameters = component_parameters
        self.hilbert_space = hilbert_space
        self.Hcs = []
        
        omega_q = component_parameters['qubit_1']['freq']
        omega_r = component_parameters['cavity']['freq']
        g = coupling_strength[0]['strength']

        dim_q = hilbert_space['qubit_1']
        dim_r  = hilbert_space['cavity']

        a = qt.tensor(qt.qeye(dim_q), qt.destroy(dim_r))
        sigmaz = qt.tensor(qt.sigmaz, qt.qeye(dim_r))
        sigmax = qt.tensor(qt.sigmax, qt.eye(dim_r))
        
        self.H0 = hbar * omega_q / 2 * sigmaz() + hbar * omega_r * a.dag() * a \
                + hbar * g * (a.dag() + a) * sigmax()
        H1 = hbar * sigmax()
        self.Hcs.append(H1)
        
        self.projector = utils.rect_space(H[0].dims[0], comp_dims) #rect identity for computation

    #TODO Think about the distinction between System and Model classes


    def get_Hamiltonian(control_fields):
        H = [H0]
        for ii in range(len(control_fields)):
            H.append([self.Hcs[ii], control_fields[ii]])
        return H
        

