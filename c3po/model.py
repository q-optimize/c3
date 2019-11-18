"""The model class, containing information on the system and its modelling."""

import numpy as np
import tensorflow as tf
from c3po.hamiltonians import resonator, duffing
from c3po.component import Qubit, Resonator, Drive, Coupling
from c3po.constants import kb, hbar


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
    H0: :class: Drift Hamiltonian
    Hcs: :class: Instruction Hamiltonians
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
        Returns the Hamiltonian
    get_time_slices()

    """

    def __init__(
            self,
            chip_elements
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
            a = np.diag(np.sqrt(np.arange(1, self.dims[indx])), k=1)
            for indy in range(len(self.dims)):
                qI = np.identity(self.dims[indy])
                if indy < indx:
                    a = np.kron(qI, a)
                if indy > indx:
                    a = np.kron(a, qI)
            self.ann_opers.append(a)

        # Create drift Hamiltonian matrices and model parameter vector
        self.params = []
        self.params_desc = []
        self.drift_Hs = []
        # TODO avoid checking element type, instead call function in element
        for element in chip_elements:
            if isinstance(element, Qubit) or isinstance(element, Resonator):
                el_indx = self.names.index(element.name)
                ann_oper = self.ann_opers[el_indx]

                self.drift_Hs.append(
                    tf.constant(
                        resonator(ann_oper),
                        dtype=tf.complex128
                    )
                )
                self.params.append(element.values['freq'])
                self.params_desc.append((element.name, 'freq'))

                if isinstance(element, Qubit) and element.hilbert_dim > 2:
                    self.drift_Hs.append(
                        tf.constant(
                            duffing(ann_oper),
                            dtype=tf.complex128
                        )
                    )
                    self.params.append(element.values['anhar'])
                    self.params_desc.append((element.name, 'anhar'))

            elif isinstance(element, Coupling):
                el_indxs = []
                for connected_element in element.connected:
                    el_indxs.append(self.names.index(connected_element))
                ann_opers = [self.ann_opers[el_indx] for el_indx in el_indxs]

                self.drift_Hs.append(
                    tf.constant(
                        element.hamiltonian(ann_opers),
                        dtype=tf.complex128
                    )
                )
                self.params.append(element.values['strength'])
                self.params_desc.append((element.name, 'strength'))

            elif isinstance(element, Drive):
                # TODO order drives by driveline name
                el_indxs = []
                h = tf.zeros(self.ann_opers[0].shape, dtype=tf.complex128)
                for connected_element in element.connected:
                    a = self.ann_opers[self.names.index(connected_element)]
                    h += tf.constant(
                        element.hamiltonian(a),
                        dtype=tf.complex128
                    )

                self.control_Hs.append(h)

        self.n_params = len(self.params)
        self.params = np.array(self.params)

    def initialise_lindbladian(self):
        """Construct Lindbladian (collapse) operators."""
        self.collapse_ops = []
        self.cops_params = []
        self.cops_params_desc = []
        self.cops_params_fcts = []

        for element in self.chip_elements:
            if isinstance(element, Qubit) or isinstance(element, Resonator):
                vals = element.values

                if 't1' in vals:
                    el_indx = self.names.index(element.name)
                    ann_oper = self.ann_opers[el_indx]
                    L1 = ann_oper

                    if 'temp' not in vals:
                        def t1(t1, L1):
                            gamma = tf.cast((0.5/t1)**0.5, tf.complex128)
                            return gamma * L1

                        self.collapse_ops.append(L1)
                        self.cops_params.append(vals['t1'])
                        self.cops_params_desc.append((element.name, 't1'))
                        self.cops_params_fcts.append(t1)

                    else:
                        L2 = ann_oper.T.conj()
                        dim = element.hilbert_dim
                        omega_q = vals['freq']
                        if 'anhar' in vals:
                            anhar = vals['anhar']
                        else:
                            anhar = 0
                        freq_diff = np.array(
                         [(omega_q + n*anhar) for n in range(dim)]
                         )

                        def t1_temp(t1_temp, L2):
                            gamma = tf.cast(
                                (0.5/t1_temp[0])**0.5, tf.complex128
                            )
                            beta = tf.cast(
                                    1 / (t1_temp[1] * kb),
                                    tf.complex128)
                            det_bal = tf.exp(-hbar*freq_diff*beta)
                            det_bal_mat = tf.linalg.tensor_diag(det_bal)
                            return gamma * (L1 + L2 @ det_bal_mat)

                        self.collapse_ops.append(L2)
                        self.cops_params.append([vals['t1'], vals['temp']])
                        self.cops_params_desc.append(
                            [element.name, 't1 & temp']
                        )
                        self.cops_params_fcts.append(t1_temp)

                if 't2star' in vals:
                    el_indx = self.names.index(element.name)
                    ann_oper = self.ann_opers[el_indx]
                    L_dep = 2 * ann_oper.T.conj() @ ann_oper

                    def t2star(t2star, L_dep):
                        gamma = tf.cast((0.5/t2star)**0.5, tf.complex128)
                        return gamma * L_dep

                    self.collapse_ops.append(L_dep)
                    self.cops_params.append(vals['t2star'])
                    self.cops_params_desc.append((element.name, 't2star'))
                    self.cops_params_fcts.append(t2star)

        self.cops_n_params = len(self.cops_params)
        self.cops_params = np.array(self.cops_params)

    def get_lindbladian(self, cops_params=None):
        """Return Lindbladian operators and their prefactors."""
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

    def get_Hamiltonians(self, params=None):
        if params is None:
            params = self.params

        drift_H = tf.zeros_like(self.drift_Hs[0])
        for ii in range(self.n_params):
            drift_H += \
                tf.cast(params[ii], tf.complex128) * self.drift_Hs[ii]

        return drift_H, self.control_Hs

    def get_parameters(self):
        values = []
        values.extend(self.params)
        if hasattr(self, 'collapse_ops'):
            values.extend(self.cops_params)
        return values

    def set_parameters(self, values):
        ln = len(self.params)
        self.params = values[:ln]
        if hasattr(self, 'collapse_ops'):
            self.cops_params = values[ln:]
        return values

    def list_parameters(self):
        par_list = []
        par_list.extend(self.params_desc)
        if hasattr(self, 'collapse_ops'):
            par_list.extend(self.cops_params_desc)
        return par_list
