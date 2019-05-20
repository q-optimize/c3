import qutip as qt
from c3po import utils


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
    def __init__(self, component_parameters, coupling, hilbert_space,
                 model_types):
        hbar = 1

        self.component_parameters = component_parameters
        self.hilbert_space = hilbert_space
        self.Hcs = []

        omega_q = component_parameters['qubit_1']['freq']
        delta = component_parameters['qubit_1']['delta']
        omega_r = component_parameters['cavity']['freq']
        g = coupling['q1_cav']['strength']

        dim_q = hilbert_space['qubit_1']
        dim_r = hilbert_space['cavity']

        res_type = model_types['cavity']
        qubit_type = model_types['qubit_1']
        inter_type = model_types['interaction']
        drive_type = model_types['direct']

        # Construct H0 from resonator, qubit and interaction types
        a = qt.tensor(qt.qeye(dim_q), qt.destroy(dim_r))
        # Resonator
        if res_type == 'harmonic':
            res = utils.hamiltonians.resonator(a, omega_r)
        # Qubit
        b = qt.tensor(qt.qeye(dim_q), qt.destroy(dim_r))
        if qubit_type == 'multi':
            qubit = utils.hamiltonians.duffing(b, omega_q, delta)
        elif qubit_type == 'simple':
            sigmaz = b * b.dag() - b.dag() * b
            qubit = omega_q / 2 * sigmaz
        # Interaction
        if inter_type == 'XX':
            inter = utils.hamiltonians.int_XX(a, b, g)
        if inter_type == 'JC':
            inter = utils.hamiltonians.int_jaynes_cummings(a, b, g)
        self.H0 = hbar * (qubit + res + inter)

        # Construct drive Hamiltonians
        if drive_type == 'direct':
            drive = hbar * (b.dag() + b)
        elif drive_type == 'indirect':
            drive = hbar * (a.dag() + a)
        self.Hcs.append(drive)

    # TODO: Think about the distinction between System and Model classes
    """
    Federico: I believe the information about the physical system,
    i.e. components and companent parameters should be in the system class
    Then the Hamiltonian is constructed in the model class with parsers
    (as above) or just provided by the user
    """

    def get_Hamiltonian(self, control_fields):
        H = [self.H0]
        for ii in range(len(control_fields)):
            H.append([self.Hcs[ii], control_fields[ii]])
        return H
