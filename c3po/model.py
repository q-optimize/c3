"""The model class, containing information on the system and its modelling."""

import numpy as np
import tensorflow as tf
from c3po.component import Drive, Coupling

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

    def __init__(self, subsystems, couplings, tasks):
        self.dressed = False
        self.lindbladian = False
        self.use_FR = True
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
        state_labels = []
        for subs in subsystems:
            dims.append(subs.hilbert_dim)
            names.append(subs.name)
            state_labels.append(list(range(subs.hilbert_dim)))
        self.tot_dim = np.prod(dims)

        self.state_labels = list(itertools.product(*state_labels))

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

        self.update_model()

        self.tasks = {}
        for task in tasks:
            self.tasks[task.name] = task

    def set_dressed(self, dressed):
        self.dressed = dressed

    def set_lindbladian(self, lindbladian):
        self.lindbladian = lindbladian

    def set_FR(self, use_FR):
        self.use_FR = use_FR

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
        if self.lindbladian:
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
        self.dressed_drift_H = dressed_drift_H
        self.dressed_control_Hs = dressed_control_Hs
        if self.lindbladian:
            for col_op in self.col_ops:
                dressed_col_ops.append(
                    tf.matmul(tf.matmul(
                        tf.linalg.adjoint(self.transform),
                        col_op),
                        self.transform
                    )
                )
            self.dressed_col_ops = dressed_col_ops

    def get_Frame_Rotation(
        self,
        t_final: np.float64,
        lo_freqs: dict,
        framechanges: dict
    ):
        # lo_freqs need to be ordered the same as the names of the qubits
        ones = tf.ones(self.tot_dim, dtype=tf.complex128)
        FR = tf.linalg.diag(ones)
        for line in lo_freqs.keys():
            lo_freq = lo_freqs[line]
            framechange = framechanges[line]
            qubit = self.couplings[line].connected[0]
            ann_oper = self.ann_opers[qubit]
            num_oper = tf.constant(
                np.matmul(ann_oper.T.conj(), ann_oper),
                dtype=tf.complex128
            )
            FR = FR * tf.linalg.expm(
                1.0j * num_oper * (lo_freq * t_final + framechange)
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
