"""The model class, containing information on the system and its modelling."""

import numpy as np
import copy
import tensorflow as tf
from scipy.linalg import expm
from c3po.hamiltonians import resonator, duffing
from c3po.component import Quantity, Qubit, Resonator, Drive, Coupling, \
    PhysicalComponent
from c3po.constants import kb, hbar
from c3po.tf_utils import tf_expm
from c3po.qt_utils import basis


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
            components
            ):

        self.spam_params = []
        self.spam_params_desc = []
        self.components = components
        self.control_Hs = []

        # Construct array with dimension of elements (only qubits & resonators)
        dims = []
        names = []
        for element in components:
            if isinstance(element, PhysicalComponent):
                dims.append(element.hilbert_dim)
                names.append(element.name)
        self.tot_dim = np.prod(dims)


        # Create anninhilation operators for physical elements
        ann_opers = []
        for indx in range(len(dims)):
            a = np.diag(np.sqrt(np.arange(1, dims[indx])), k=1)
            for indy in range(len(dims)):
                qI = np.identity(dims[indy])
                if indy < indx:
                    a = np.kron(qI, a)
                if indy > indx:
                    a = np.kron(a, qI)
            ann_opers.append(a)

        self.dims = {}
        self.ann_opers = {}
        for indx in range(len(dims)):
self.dims[name[indx]] = dims[indx]
            self.ann_opers[name[indx]] = ann_opers[indx]


        # Create drift Hamiltonian matrices and model parameter vector
        # TODO avoid checking element type, instead call function in element
        for element in components:
            if isinstance(element, PhysicalComponent):
                element.init_Hs(self.ann_opers[element.name])

            elif isinstance(element, LineComponent):
                element.init_Hs(
                    [self.ann_opers for connected_element in element.connected]
                )
            # TODO order drives by driveline name

    def write_config(self):
        return "We don't care about the model... YET!"

    def initialise_lindbladian(self):
        """Construct Lindbladian (collapse) operators."""
        self.collapse_ops = []
        self.cops_params = []
        self.cops_params_desc = []
        self.cops_params_fcts = []

        for element in self.components:
            if isinstance(element, Qubit) or isinstance(element, Resonator):
                vals = element.values
                el_indx = self.names.index(element.name)
                ann_oper = self.ann_opers[el_indx]

                if 't1' in vals:

                    L1 = ann_oper

                    if 'temp' not in vals:
                        def t1(t1, L1):
                            gamma = (0.5 / t1.tf_get_value()) ** 0.5
                            return gamma * L1

                        self.collapse_ops.append(L1)
                        self.cops_params.append(vals['t1'])
                        self.cops_params_desc.append((element.name, 't1'))
                        self.cops_params_fcts.append(t1)

                    else:
                        L2 = ann_oper.T.conj()
                        dim = element.hilbert_dim
                        omega_q = vals['freq'].tf_get_value()
                        if 'anhar' in vals:
                            anhar = vals['anhar'].tf_get_value()
                        else:
                            anhar = 0
                        # TODO This breaks tensorflow for temp
                        freq_diff = np.array(
                         [(omega_q + n*anhar) for n in range(dim)]
                         )

                        def t1_temp(t1_temp, L2):
                            gamma = (0.5/t1_temp[0].tf_get_value())**0.5
                            beta = 1 / (t1_temp[1].tf_get_value() * kb)
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
                        gamma = (0.5/t2star.tf_get_value())**0.5
                        return gamma * L_dep

                    self.collapse_ops.append(L_dep)
                    self.cops_params.append(vals['t2star'])
                    self.cops_params_desc.append((element.name, 't2star'))
                    self.cops_params_fcts.append(t2star)

        self.cops_n_params = len(self.cops_params)

    def get_lindbladian(self, cops_params=None):
        """Return Lindbladian operators and their prefactors."""
        if cops_params is None:
            cops_params = self.cops_params

        col_ops = []
        for ii in range(self.cops_n_params):
            col_ops.append(
                tf.cast(
                    self.cops_params_fcts[ii](
                        self.cops_params[ii], self.collapse_ops[ii]
                    ), tf.complex128
                )
            )

        if self.dress:
            for indx in range(len(col_ops)):
                col_ops[indx] = tf.matmul(
                    tf.matmul(tf.linalg.adjoint(transform), col_ops[indx]),
                    transform
                )

        return col_ops

    def get_Hamiltonians(self, params=None):
        if params is None:
            params = self.params

        drift_H = tf.zeros_like(self.drift_Hs[0])
        for ii in range(self.n_params):
            drift_H += tf.cast(
                params[ii].tf_get_value(), tf.complex128
            ) * self.drift_Hs[ii]
        control_Hs = self.control_Hs

        if self.dress:
            drift_H = tf.matmul(
                tf.matmul(tf.linalg.adjoint(transform), drift_H),
                transform
            )
            for indx in range(len(control_Hs)):
                control_Hs[indx] = tf.matmul(
                    tf.matmul(tf.linalg.adjoint(transform), control_Hs[indx]),
                    transform
                )

        return drift_H, control_Hs

    def get_Virtual_Z(self, t_final, freqs):
        # lo_freqs need to be ordered the same as the names of the qubits
        anns = []
        # freqs = []
        for name in self.names:
            # TODO Effectively collect parameters of the virtual Z
            if name[0] == 'q' or name[0] == 'Q':
                ann_indx = self.names.index(name)
                anns.append(self.ann_opers[ann_indx])
                # freq_indx = self.params_desc.index((name, 'freq'))
                # freqs.append(self.params[freq_indx])

        # TODO make sure terms is right
        # num_oper = np.matmul(anns[0].T.conj(), anns[0])
        num_oper = tf.constant(
            np.matmul(anns[0].T.conj(), anns[0]),
            dtype=tf.complex128
        )
        VZ = tf.linalg.expm(1.0j * num_oper * (freqs[0] * t_final))
        for ii in range(1, len(anns)):
            num_oper = tf.constant(
                np.matmul(anns[ii].T.conj(), anns[ii]),
                dtype=tf.complex128
            )
            VZ = VZ * tf.linalg.expm(1.0j * num_oper * (freqs[ii] * t_final))

        if self.dress:
            VZ = tf.matmul(
                tf.matmul(tf.linalg.adjoint(self.transform), VZ),
                self.transform
            )
        return VZ

    def get_drift_eigen(self, params=None, ordered=True):
        if params is None:
            params = self.params

        drift_H = tf.zeros_like(self.drift_Hs[0])
        for ii in range(self.n_params):
            drift_H += tf.cast(
                params[ii].tf_get_value(), tf.complex128
            ) * self.drift_Hs[ii]

        e, v = tf.linalg.eigh(drift_H)

        if ordered:
            reorder_matrix = tf.cast(tf.round(tf.abs(v)), tf.complex128)
            e = tf.reshape(e, [e.shape[0], 1])
            eigenframe = tf.matmul(
                reorder_matrix, e
            )
            # tmp = tf.matmul(reorder_matrix, v)
            transform = tf.matmul(v, reorder_matrix)
            # order = tf.argmax(tf.abs(v), axis=0)
            # np_transform = np.zeros_like(drift_H.numpy())
            # np_diag = np.zeros_like(e.numpy())
            # for count in range(len(e)):
            #     indx = order[count]
            #     np_transform[:,indx] = v[:,count].numpy()
            #     np_diag[indx] = e[count]
            # transform = tf.constant(np_transform, dtype=tf.complex128)
            # diag = tf.constant(np_diag, dtype=tf.complex128)
            # eigenframe = tf.linalg.diag(diag)
        else:
            eigenframe = tf.linalg.diag(e)
            transform = v

        return eigenframe, transform

    def recalc_dressed(self):
        self.eigenframe, self.transform = get_drift_eigen()

    def get_qubit_freqs(self):
        # TODO figure how to get the correct dressed frequencies
        pass


    # things that deal with parameters

    def get_parameters(self, scaled=False):
        values = []
        for par in self.params:
            if scaled:
                values.append(par.value.numpy())
            else:
                values.append(par.numpy())
        if hasattr(self, 'collapse_ops'):
            for par in self.cops_params:
                if scaled:
                    values.append(par.value.numpy())
                else:
                    values.append(par.numpy())
        for par in self.spam_params:
            if scaled:
                values.append(par.value.numpy())
            else:
                values.append(par.numpy())
        return values

    def set_parameters(self, values):
        ln = len(self.params)
        ln_s = len(values)-len(self.spam_params)
        for ii in range(0, ln):
            self.params[ii].tf_set_value(values[ii])
        if hasattr(self, 'collapse_ops'):
            for ii in range(ln, ln_s):
                self.cops_params[ii-ln].tf_set_value(values[ii])
        for ii in range(ln_s,len(values)):
            self.spam_params[ii-ln_s].tf_set_value(values[ii])
        self.recalc_dressed()

    def list_parameters(self):
        par_list = []
        par_list.extend(self.params_desc)
        if hasattr(self, 'collapse_ops'):
            par_list.extend(self.cops_params_desc)
        par_list.extend(self.spam_params_desc)
        return par_list

    # From here there is temporary code that deals with initialization and
    # measurement

    @staticmethod
    def populations(state, lindbladian):
        if lindbladian:
            diag = []
            dim = int(tf.sqrt(len(state)))
            indeces = [n * dim + n for n in range(dim)]
            for indx in indeces:
                diag.append(state[indx])
            return tf.abs(diag)
        else:
            return tf.abs(state)**2

    def percentage_01_spam(self, state, lindbladian):
        indx_ms = self.spam_params_desc.index('meas_offset')
        indx_im = self.spam_params_desc.index('initial_meas')
        meas_offsets = self.spam_params[indx_ms].tf_get_value()
        initial_meas = self.spam_params[indx_im].tf_get_value()
        row1 = initial_meas + meas_offsets
        row1 = tf.reshape(row1, [1, row1.shape[0]])
        extra_dim = int(len(state)/len(initial_meas))
        if extra_dim != 1:
            row1 = tf.concat([row1]*extra_dim, 1)
        row2 = tf.ones_like(row1) - row1
        conf_matrix = tf.concat([row1, row2], 0)
        pops = self.populations(state, lindbladian)
        pops = tf.reshape(pops, [pops.shape[0],1])
        return tf.matmul(conf_matrix, pops)

    def set_spam_param(self, name: str, quan: Quantity):
        self.spam_params.append(quan)
        self.spam_params_desc.append(name)

    def initialise(self):
        indx_it = self.spam_params_desc.index('init_temp')
        init_temp = self.spam_params[indx_it].tf_get_value()
        init_temp = tf.cast(init_temp, dtype=tf.complex128)
        drift_H, control_Hs = self.get_Hamiltonians()
        # diag = tf.math.real(tf.linalg.diag_part(drift_H))
        diag = tf.linalg.diag_part(drift_H)
        freq_diff = diag - diag[0]
        beta = 1 / (init_temp * kb)
        det_bal = tf.exp(-hbar * freq_diff * beta)
        norm_bal = det_bal / tf.reduce_sum(det_bal)
        return tf.reshape(tf.sqrt(norm_bal), [norm_bal.shape[0],1])
