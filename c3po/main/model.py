import qutip as qt

from c3po import utils


class Model:
    """
    What the theorist thinks about from the system.
    Class to store information about our system/problem/device. Different
    models can represent the same system.

    Parameters
    ---------
    component_parameters : dict of dict
    couplings : dict of dict
    hilbert_space : dict
        Hilbert space dimensions of full space
    comp_hilbert_space : dict
        Hilbert space dimensions of computational space

    Attributes
    ----------
    H0: :class:'qutip.qobj' Drift Hamiltonian
    Hcs: :class:'list of qutip.qobj' Control Hamiltonians

    component_parameters :

    control_fields: list
        [args, func1_t, func2_t, ,...]

    coupling :

    hilbert_space :

    Methods
    -------
    construct_Hamiltonian(component_parameters, hilbert_space)
        Construct a model for this system, to be used in numerics.
    get_Hamiltonian()
        Returns the Hamiltonian in a QuTip compatible way
    get_time_slices()
    """
    def __init__(self, component_parameters, coupling, hilbert_space,tf_flag="False"):

        hbar = 1

        self.component_parameters = component_parameters
        self.coupling = coupling
        self.hilbert_space = hilbert_space
        self.Hcs = []

        omega_q = component_parameters['qubit_1']['freq']
        omega_r = component_parameters['cavity']['freq']
        g = coupling['q1_cav']['strength']

        dim_q = hilbert_space['qubit_1']
        dim_r = hilbert_space['cavity']

        a = qt.tensor(qt.qeye(dim_q), qt.destroy(dim_r))
        sigmaz = qt.tensor(qt.sigmaz(), qt.qeye(dim_r))
        sigmax = qt.tensor(qt.sigmax(), qt.qeye(dim_r))

        self.H0 = hbar * omega_q / 2 * sigmaz + hbar * omega_r * a.dag() * a \
            + hbar * g * (a.dag() + a) * sigmax
        H1 = hbar * sigmax
        self.Hcs.append(H1)


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

