"""The model class, containing information on the system and its modelling."""

import numpy as np
import tensorflow as tf
from c3po.component import Quantity, Drive, Coupling
from c3po.constants import kb, hbar


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

    def __init__(self, subsystems, couplings):
        self.dressed = False
        self.params = {}
        self.subsystems = {}
        for comp in subsystems:
            self.subsystems[comp.name] = comp
        self.couplings = {}
        for comp in couplings:
            self.couplings[comp.name] = comp

        # Construct array with dimension of comps (only qubits & resonators)
        dims = []
        names = []
        for subs in subsystems:
            dims.append(subs.hilbert_dim)
            names.append(subs.name)
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
            self.dims[names[indx]] = dims[indx]
            self.ann_opers[names[indx]] = ann_opers[indx]

        # Create drift Hamiltonian matrices and model parameter vector
        for subs in subsystems:
            ann_oper = self.ann_opers[subs.name]
            subs.init_Hs(ann_oper)
            subs.init_Ls(ann_oper)

        for line in couplings:
            line.init_Hs(self.ann_opers)
        dim = sum(self.dims.values())

        self.update_model()

    def list_parameters(self):
        ids = []
        for key in self.params:
            ids.append(("Model", key))
        return ids

    def get_Hamiltonians(self):
        if self.dressed:
            return self.dressed_drift_H, self.dressed_control_Hs
        else:
            return self.drift_H, self.control_Hs

    def get_Lindbladians(self):
        if self.dressed:
            return self.dressed_col_ops
        else:
            return self.col_ops

    def update_model(self):
        self.update_Hamiltonians()
        self.update_Lindbladians()
        if self.dressed:
            self.update_dressed()

    def update_Hamiltonians(self):
        control_Hs = {}
        drift_H = tf.zeros([self.tot_dim, self.tot_dim], dtype=tf.complex128)
        for sub in self.subsystems.values():
            drift_H += sub.get_Hamiltonian()
        for key, line in self.couplings.items():
            if isinstance(line, Coupling):
                drift_H += line.get_Hamiltonian()
            elif isinstance(line, Drive):
                control_Hs[key] = line.get_Hamiltonian()
        self.drift_H = drift_H
        self.control_Hs = control_Hs

    def update_Lindbladians(self):
        """Return Lindbladian operators and their prefactors."""
        col_ops = []
        for subs in self.subsystems.values():
            col_ops.append(subs.get_Lindbladian())
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
        self.update_drift_eigen()
        dressed_control_Hs = {}
        dressed_col_ops = []
        dressed_drift_H = tf.matmul(tf.matmul(
            tf.linalg.adjoint(self.transform),
            self.drift_H),
            self.transform
        )
        for key in self.control_Hs:
            dressed_control_Hs[key] = tf.matmul(tf.matmul(
                tf.linalg.adjoint(self.transform),
                self.control_Hs[key]),
                self.transform
            )
        for col_op in self.col_ops:
            dressed_col_ops.append(
                tf.matmul(tf.matmul(
                    tf.linalg.adjoint(self.transform),
                    col_op),
                    self.transform
                )
            )
        self.dressed_drift_H = dressed_drift_H
        self.dressed_control_Hs = dressed_control_Hs
        self.dressed_col_ops = dressed_col_ops

    def get_Frame_Rotation(
        self,
        t_final: np.float64,
        lo_freqs: dict
    ):
        # lo_freqs need to be ordered the same as the names of the qubits
        ones = tf.ones(self.tot_dim, dtype=tf.complex128)
        FR = tf.linalg.diag(ones)
        for line, lo_freq in lo_freqs.items():
            qubit = self.couplings[line].connected[0]
            ann_oper = self.ann_opers[qubit]
            num_oper = tf.constant(
                np.matmul(ann_oper.T.conj(), ann_oper),
                dtype=tf.complex128
            )
            FR = FR * tf.linalg.expm(
                1.0j * num_oper * lo_freq * t_final
            )
        if self.dressed:
            FR = tf.matmul(tf.matmul(
                tf.linalg.adjoint(self.transform),
                FR),
                self.transform
            )
        return FR

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

    def pop1_spam(self, state, lindbladian):
        if 'confusion_row' in self.params:
            row1 = self.params['confusion_row'].get_value()
            row2 = tf.ones_like(row1) - row1
            conf_matrix = tf.concat([[row1], [row2]], 0)
        elif 'confusion_matrix' in self.params:
            conf_matrix = self.params['confusion_matrix'].get_value()
        pops = self.populations(state, lindbladian)
        pops = tf.reshape(pops, [pops.shape[0], 1])
        pop1 = tf.matmul(conf_matrix, pops)[1]
        if 'meas_offset' in self.params:
            pop1 = pop1 - self.params['meas_offset'].get_value()
        if 'meas_scale' in self.params:
            pop1 = pop1 * self.params['meas_scale'].get_value()
        return pop1

    def set_spam_param(self, name: str, quan: Quantity):
        self.params[name] = quan

    def initialise(self):
        init_temp = tf.cast(
            self.params['init_temp'].get_value(), dtype=tf.complex128
        )
        diag = tf.linalg.diag_part(self.dressed_drift_H)
        freq_diff = diag - diag[0]
        beta = 1 / (init_temp * kb)
        det_bal = tf.exp(-hbar * freq_diff * beta)
        norm_bal = det_bal / tf.reduce_sum(det_bal)
        return tf.reshape(tf.sqrt(norm_bal), [norm_bal.shape[0], 1])
