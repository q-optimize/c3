"""The model class, containing information on the system and its modelling."""

import numpy as np
import copy
import tensorflow as tf
from scipy.linalg import expm
from c3po.hamiltonians import resonator, duffing
from c3po.component import Quantity, Qubit, Resonator, Drive, Coupling
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
    hilbert_space : dict
        Hilbert space dimensions of full space


    Attributes
    ----------
    H0: :class: Drift Hamiltonian


    Methods
    -------
    construct_Hamiltonian(component_parameters, hilbert_space)
        Construct a model for this system, to be used in numerics.

    """

    def __init__(
            self,
            phys_components,
            line_components
            ):

        self.phys_components = phys_components
        self.line_components = line_components

        # Construct array with dimension of comps (only qubits & resonators)
        dims = []
        names = []
        for phys_comp in phys_components:
            dims.append(phys_comp.hilbert_dim)
            names.append(phys_comp.name)
        self.tot_dim = np.prod(dims)

        # Create anninhilation operators for physical comps
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
        for phys_comp in phys_components:
            ann_oper = self.ann_opers[comp.name]
            phys_comp.init_Hs(ann_oper)
            phys_comp.init_Ls(ann_oper)

        for line_comp in line_components:
            ann_opers = [
                self.ann_opers[conn_comp] for conn_comp in comp.connected
            ]
            comp.init_Hs(ann_opers)
            # TODO order drives by driveline name

        self.update_model()

    def update_model(self):
        self.update_Hamiltonians()
        self.update_Lindbladians()
        self.update_drift_eigen()
        self.update_dressed()

    def update_Hamiltonians(self):
        control_Hs = []
        drift_H = tf.zeros(self.tot_dim)
        for phys_comp in self.phys_components:
            drift_H += phys_comp.get_Hamiltonian()

        for line_comp in self.line_components:
            if isinstance(line_comp, Coupling):
                drift_H += line_comp.get_Hamiltonian
            elif isinstance(line_comp, Drive):
                control_Hs.append(line_comp.get_Hamiltonian)

    def update_Lindbladians(self):
        """Return Lindbladian operators and their prefactors."""
        col_ops = []
        for phys_comp in self.phys_components:
            col_ops.append(phys_comp.get_Lindbladian())

        if self.dress:
            for indx in range(len(col_ops)):
                col_ops[indx] = tf.matmul(tf.matmul(
                    tf.linalg.adjoint(self.transform),
                    col_ops[indx]),
                    self.transform
                )

        self.col_ops = col_ops

    def update_drift_eigen(self, ordered=True):
        e, v = tf.linalg.eigh(self.drift_H)

        if ordered:
            reorder_matrix = tf.cast(tf.round(tf.abs(v)), tf.complex128)
            e = tf.reshape(e, [e.shape[0], 1])
            eigenframe = tf.matmul(reorder_matrix, e)
            transform = tf.matmul(v, reorder_matrix)
        else:
            eigenframe = tf.linalg.diag(e)
            transform = v

        self.eigenframe = eigenframe
        self.transform = transform

    def update_dressed(self):
        dressed_control_Hs = []
        dressed_drift_H = tf.matmul(tf.matmul(
            tf.linalg.adjoint(self.transform),
            self.drift_H),
            self.transform
        )
        for indx in range(len(control_Hs)):
            dressed_control_Hs[indx] = tf.matmul(tf.matmul(
                tf.linalg.adjoint(self.transform),
                self.control_Hs[indx]),
                self.transform
            )

        self.dressed_drift_H = dressed_drift_H
        self.dressed_control_Hs = dressed_control_Hs

    def get_Virtual_Z(self, t_final, lo_freqs: dict):
        # lo_freqs need to be ordered the same as the names of the qubits
        anns = []
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

    def get_qubit_freqs(self):
        # TODO figure how to get the correct dressed frequencies
        pass

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
