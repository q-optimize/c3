import numpy as np
import qutip as qt
import tensorflow as tf
from c3po.hamiltonians import *
from c3po.component import *

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


    def __init__(
            self,
            chip_elements,
            mV_to_Hz,
            ):

        self.chip_elements = chip_elements
        self.control_Hs = []


        # Construct array with dimension of elements (only qubits & resonators)
        self.dims = []
        self.names = []

        for element in chip_elements:

            if isinstance(element, Qubit) or isinstance(element, Resonator):
                self.dims.append(element.hilbert_dim)
                self.names.append(element.name)

        # Create anninhilation operators for physical elements
        self.ann_opers = []

        for indx in range(len(self.dims)):
            a = qt.destroy(self.dims[indx])

            for indy in range(len(self.dims)):
                qI = qt.qeye(self.dims[indy])
                if indy < indx:
                    a = qt.tensor(qI, a)
                if indy > indx:
                    a = qt.tensor(a, qI)

            self.ann_opers.append(a.full())

        # Create drift Hamiltonian matrices and model parameter vector
        self.params = []
        self.params_desc = []
        self.drift_Hs = []
        for element in chip_elements:
            if isinstance(element, Qubit) or isinstance(element, Resonator):
                el_indx = self.names.index(element.name)
                ann_oper = self.ann_opers[el_indx]

                self.drift_Hs.append(resonator(ann_oper))
                self.params.append(element.values['freq'])
                self.params_desc.append([element.name, 'freq'])

                if isinstance(element, Qubit) and element.hilbert_dim > 2:
                    self.drift_Hs.append(duffing(ann_oper))
                    self.params.append(element.values['delta'])
                    self.params_desc.append([element.name, 'delta'])

            elif isinstance(element, Coupling):
                el_indxs = []

                for connected_element in element.connected:
                    el_indxs.append(self.names.index(connected_element))

                ann_opers = [self.ann_opers[el_indx] for el_indx in el_indxs]

                self.drift_Hs.append(int_XX(ann_opers))
                self.params.append(element.values['strength'])
                self.params_desc.append([element.name, 'strength'])

            elif isinstance(element, Drive):
                el_indxs = []
                h = tf.zeros(self.ann_opers[0].shape, dtype=tf.complex128)
                for connected_element in element.connected:
                    a = self.ann_opers[self.names.index(connected_element)]
                    h += element.Hamiltonian(a)

                self.control_Hs.append(h)
                self.params.append(mV_to_Hz)
                self.params_desc.append([element.name, 'response'])

        self.n_params = len(self.params)
        self.params = np.array(self.params)

    def initialise_lindbladian(self):
        boltzmann = 1.380649e-23
        h = 6.62607015e-34
        hbar = h / (2*np.pi)

        self.collapse_ops = []
        self.cops_params = []
        self.cops_params_desc = []
        self.cops_params_fcts = []


        for element in self.chip_elements:
            vals = element.values

            if 'T1' in vals:
                el_indx = self.names.index(element.name)
                ann_oper = self.ann_opers[el_indx]
                L1 = ann_oper

                if 'temp' not in vals:
                    def T1(T1, L1):
                        gamma = tf.cast((0.5/T1)**0.5, tf.complex128)
                        return gamma * L1

                    self.collapse_ops.append(L1)
                    self.cops_params.append(vals['T1'])
                    self.cops_params_desc.append([element.name, 'T1'])
                    self.cops_params_fcts.append(T1)

                else:
                    L2 = ann_oper.T.conj()
                    dim = element.hilbert_dim
                    omega_q = vals['freq']
                    if 'delta' in vals: delta = vals['delta']
                    else: delta = 0
                    freq_diff = np.array(
                     [(omega_q + n*delta) for n in range(dim)]
                     )
                    def T1_temp(T1_temp, L2):
                        gamma = tf.cast((0.5/T1_temp[0])**0.5, tf.complex128)
                        beta = tf.cast(
                                1 / (T1_temp[1] * boltzmann),
                                tf.complex128)
                        det_bal = tf.exp(-hbar*freq_diff*beta)
                        det_bal_mat = tf.linalg.tensor_diag(det_bal)
                        return gamma * (L1 + L2 @ det_bal_mat)

                    self.collapse_ops.append(L2)
                    self.cops_params.append([vals['T1'],vals['temp']])
                    self.cops_params_desc.append([element.name, 'T1 & temp'])
                    self.cops_params_fcts.append(T1_temp)

            if 'T2star' in vals:
                el_indx = self.names.index(element.name)
                ann_oper = self.ann_opers[el_indx]
                L_dep = 2 * ann_oper.T.conj() @ ann_oper
                def T2star(T2star, L_dep):
                    gamma = tf.cast((0.5/T2star)**0.5, tf.complex128)
                    return gamma * L_dep

                self.collapse_ops.append(L_dep)
                self.cops_params.append(vals['T2star'])
                self.cops_params_desc.append([element.name, 'T2star'])
                self.cops_params_fcts.append(T2star)

        self.cops_n_params = len(self.cops_params)
        self.cops_params = np.array(self.cops_params)

    def get_lindbladian(self, cops_params=None):
        if cops_params is None:
            cops_params = self.cops_params

        col_ops = []
        for ii in range(self.cops_n_params):
            col_ops.append(
                    self.cops_params_fcts[ii](
                        self.cops_params[ii],
                        self.collapse_ops[ii]
                        )
                    )
        return col_ops

    def update_parameters(self, new_params):
        idx = 0
        self.params = new_params
        for element in self.chip_elements:

            if isinstance(element, Qubit):
                element.values['freq'] = new_params[idx]
                idx += 1
                if element.hilbert_dim > 2:
                    element.values['delta'] = new_params[idx]
                    idx += 1

            elif isinstance(element, Resonator):
                element.values['freq'] = new_params[idx]
                idx += 1

            elif isinstance(element, Coupling):
                element.values['strength'] = new_params[idx]
                idx += 1

    def get_Hamiltonians(self, params=None):
        if params is None:
            params = self.params

        drift_H = tf.zeros_like(self.drift_Hs[0])
        control_Hs = []
        di = 0
        ci = 0
        for ii in range(self.n_params):

            if self.params_desc[ii][1]=='response':
                control_Hs.append(
                    tf.cast(params[ii], tf.complex128) * self.control_Hs[ci]
                    )
                ci += 1
            else:
                drift_H += tf.cast(params[ii], tf.complex128) * self.drift_Hs[di]
                di += 1

        return drift_H, control_Hs

    def get_values_bounds(self):
        values = self.params
        bounds = np.kron(np.array([[0.5], [1.5]]), self.params)
        return values, bounds.T
