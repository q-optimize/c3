"""The model class, containing information on the system and its modelling."""

import numpy as np
import copy
import tensorflow as tf
from scipy.linalg import expm
from c3po.hamiltonians import resonator, duffing
from c3po.component import Qubit, Resonator, Drive, Coupling
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
            chip_elements
            ):

        self.chip_elements = chip_elements
        self.control_Hs = []

        # Construct array with dimension of elements (only qubits & resonators)
        self.dims = []
        # TODO store also total dimension of the hilbert space
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
                # TODO change all .values to .params or viceversa
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

    def write_config(self):
        return "We don't care about the model... YET!"

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
                            gamma =(0.5/t1.tf_get_value())**0.5
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
                            gamma = (0.5/t1_temp[0])**0.5
                            beta = 1 / (t1_temp[1] * kb)
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
                        self.cops_params[ii],self.collapse_ops[ii]
                    ), tf.complex128
                )
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

        return drift_H, self.control_Hs

    def get_drift_eigenframe(self, params=None):
        if params is None:
            params = self.params

        drift_H = tf.zeros_like(self.drift_Hs[0])
        for ii in range(self.n_params):
            drift_H += \
                tf.cast(params[ii], tf.complex128) * self.drift_Hs[ii]

        e,v = tf.linalg.eigh(drift_H)
        eigenframe = tf.zeros_like(drift_H)
        eigenframe = tf.linalg.set_diag(eigenframe, e)

        order = tf.argmax(tf.abs(v), axis=1)
        np_transform = np.zeros_like(drift_H.numpy())
        for indx in order:
            np_transform[:,indx] = v[indx].numpy()
        transform = tf.constant(np_transform, dtype=tf.complex128)

        return eigenframe, transform

    def dress_Hamiltonians(self, params=None):
        if self.dressed = True:
            pass

        eigenframe, transform = self.get_drift_eigenframe(params=None)
        drift_Hs = self.drift_Hs
        control_Hs = self.control_Hs
        for indx in range(len(drift_Hs)):
            drift_h = drift_Hs[indx]
            drift_Hs[indx] = tf.matmul(
                tf.matmul(transform, drift_h),
                tf.linalg.adjoint(transform)
            )
        for indx in range(len(control_Hs)):
            ctrl_h = control_Hs[indx]
            control_Hs[indx] = tf.matmul(
                tf.matmul(transform, ctrl_h),
                tf.linalg.adjoint(transform)
            )
        self.drift_Hs = drift_Hs
        self.control_Hs = control_Hs
        self.transform = transform
        self.dressed = True
        return drift_Hs, control_Hs

    def get_Virtual_Z(self, t_final):
        anns = []
        freqs = []
        for name in self.names:
            # TODO Effectively collect parameters of the virtual Z
            if name[0] == 'q' or name[0] == 'Q':
                ann_indx = self.names.index(name)
                anns.append(self.ann_opers[ann_indx])
                freq_indx = self.params_desc.index((name, 'freq'))
                freqs.append(self.params[freq_indx])


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
        return VZ

    def get_parameters(self, scaled=False):
        values = []
        for par in self.params:
            if scaled:
                values.append(float(par.value))
            else:
                values.append(float(par))
        if hasattr(self, 'collapse_ops'):
            for par in self.cops_params:
                if scaled:
                    values.append(float(par.value))
                else:
                    values.append(float(par))
        return values

    def set_parameters(self, values):
        ln = len(self.params)
        for ii in range(0, ln):
            self.params[ii].tf_set_value(values[ii])
        if hasattr(self, 'collapse_ops'):
            for ii in range(ln, len(values)):
                self.cops_params[ii-ln].tf_set_value(values[ii])

    def list_parameters(self):
        par_list = []
        par_list.extend(self.params_desc)
        if hasattr(self, 'collapse_ops'):
            par_list.extend(self.cops_params_desc)
        return par_list

    # From here there is temporary code that deals with initialization and
    # measurement

    def readout_au_phase(self,
        factor: Quantity = None,
        offset: Quantity = None,
        phase = None
    ):
        """Fake the readout process by multiplying a state phase with a factor."""
        offset = self.spam_params['factor'].tf_get_value()
        factor = self.spam_params['offset'].tf_get_value()
        return phase * factor + offset

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

    def percentage_01_spam(state, lindbladian):
        meas_error = self.spam_params['meas_err'].tf_get_value()
        bias = self.spam_params['bias'].tf_get_value()
        pops = self.populations(state, lindbladian)

    def set_spam_param(name: str, quan: Quantity):
        self.spam_params[name] = quan

    def initialise():
        dims = tf.reduce_prod(self.dims)
        init_temp = self.spam_params['init_temp']
        # check if the dressed basis is "actived" else activate
        if self.dressed = False:
            self.dress_Hamiltonians()
        drift_H, control_Hs = self.get_Hamiltonians()
        diag = tf.linalg.diag_part(drift_H)
        freq_diff = diag[1:] - diag[:-1]
        beta = 1 / (init_temp * kb)
        det_bal = tf.exp(-hbar*freq_diff*beta)
        init_psi = basis(dims,0) * (1 - tf.reduce_sum(det_bal))
        for level in range(dims-1):
            init_psi = init_psi + basis(dims,level) * det_bal[level]
        return init_psi
