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
    def __init__(self,
                 component_parameters,
                 couplings,
                 hilbert_space,
                 comp_hilbert_space={},
                 model_types={}):

        hbar = 1

        if len(component_parameters) != len(hilbert_space) or \
                len(hilbert_space) != len(comp_hilbert_space) or \
                len(comp_hilbert_space) != len(component_parameters):
            raise ValueError('Dimensions do not match')

        self.component_parameters = component_parameters
        self.hilbert_space = hilbert_space
        self.comp_hilbert_space = comp_hilbert_space
        self.model_types = model_types
        self.Hcs = []

        # Ensure we mantain correct ordering
        self.component_keys = list(component_parameters.keys())
        # self.coupling_keys = list(couplings.keys())
        self.dims = [hilbert_space[x] for x in self.component_keys]

        # Anninhilation_operators
        ann_opers = []
        for indx in range(len(self.dims)):
            a = qt.destroy(self.dims[indx])
            for indy in range(len(self.dims)):
                qI = qt.qeye(self.dims[indy])
                if indy < indx:
                    a = qt.tensor(qI, a)
                if indy > indx:
                    a = qt.tensor(a, qI)
            ann_opers.append(a)

        if model_types:  # check if model types have been assinged
            static_Hs = []
            component_models = model_types['components']
            for component in component_models.keys():
                index = self.component_keys.index(component)
                ann_oper = ann_opers[index]
                # TODO improve check if qubit or resonator
                if component[0] == 'q':
                    hamiltonia_fun = component_models[component]
                    omega_q = component_parameters[component]['freq']
                    delta = component_parameters[component]['delta']
                    static_Hs.append(hamiltonia_fun(ann_oper, omega_q, delta))
                if component[0] == 'r':
                    hamiltonia_fun = component_models[component]
                    omega_r = component_parameters[component]['freq']
                    static_Hs.append(hamiltonia_fun(ann_oper, omega_r))
            coupling_models = model_types['couplings']
            for coupling in coupling_models.keys():
                # order is important
                index1 = self.component_keys.index(coupling[0])
                index2 = self.component_keys.index(coupling[1])
                ann_oper1 = ann_opers[index1]
                ann_oper2 = ann_opers[index2]
                g = couplings[coupling]['strength']
                hamiltonia_fun = coupling_models[coupling]
                static_Hs.append(hamiltonia_fun(ann_oper1, ann_oper2, g))
            self.H0 = hbar * sum(static_Hs)
            drive_models = model_types['drives']
            for drive in drive_models.keys():
                index = self.component_keys.index(drive)
                ann_oper = ann_opers[index]
                hamiltonia_fun = drive_models[drive]
                self.Hcs.append(hamiltonia_fun(ann_oper))

        else:
            # ###Old version###

            omega_q = component_parameters['qubit_1']['freq']
            delta = component_parameters['qubit_1']['delta']
            omega_r = component_parameters['cavity']['freq']
            g = couplings['q1_cav']['strength']

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
