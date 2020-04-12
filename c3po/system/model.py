"""The model class, containing information on the system and its modelling."""

import numpy as np
import itertools
import tensorflow as tf
import c3po.utils.tf_utils as tf_utils
import c3po.utils.qt_utils as qt_utils
from c3po.system.chip import Drive, Coupling

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
        self.dephasing_strength = 0.0
        self.params = {}
        self.subsystems = {}
        for comp in subsystems:
            self.subsystems[comp.name] = comp
        self.couplings = {}
        for comp in couplings:
            self.couplings[comp.name] = comp

        # HILBERT SPACE
        dims = []
        names = []
        state_labels = []
        comp_state_labels = []
        for subs in subsystems:
            dims.append(subs.hilbert_dim)
            names.append(subs.name)
            state_labels.append(list(range(subs.hilbert_dim)))
            comp_state_labels.append([0, 1])
        self.tot_dim = np.prod(dims)
        self.names = names
        self.state_labels = list(itertools.product(*state_labels))
        self.comp_state_labels = list(itertools.product(*comp_state_labels))

        # Create anninhilation operators for physical comps
        ann_opers = []
        for indx in range(len(dims)):
            a = np.diag(np.sqrt(np.arange(1, dims[indx])), k=1)
            ann_opers.append(
                qt_utils.hilbert_space_kron(a, indx, dims)
            )

        self.dims = dims
        self.ann_opers = ann_opers

        # Create drift Hamiltonian matrices and model parameter vector
        indx = 0
        for subs in subsystems:
            subs.init_Hs(ann_opers[indx])
            subs.init_Ls(ann_opers[indx])
            subs.set_subspace_index(indx)
            indx += 1

        for line in couplings:
            conn = line.connected
            opers_list = []
            for sub in conn:
                indx = names.index(sub)
                opers_list.append(self.ann_opers[indx])
            line.init_Hs(opers_list)

        self.update_model()

        self.tasks = {}
        for task in tasks:
            self.tasks[task.name] = task

    def set_dressed(self, dressed):
        self.dressed = dressed
        self.update_model()

    def set_lindbladian(self, lindbladian):
        self.lindbladian = lindbladian
        self.update_model()

    def set_FR(self, use_FR):
        self.use_FR = use_FR

    def set_dephasing_strength(self, dephasing_strength):
        self.dephasing_strength = dephasing_strength

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
        tot_dim = self.tot_dim
        drift_H = tf.zeros([tot_dim, tot_dim], dtype=tf.complex128)
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
            col_ops.append(subs.get_Lindbladian(self.dims))
        self.col_ops = col_ops

    def update_drift_eigen(self, ordered=True):
        e, v = tf.linalg.eigh(self.drift_H)
        reorder_matrix = tf.cast(tf.round(tf.math.abs(v)), tf.complex128)
        if ordered:
            eigenframe = tf.linalg.matvec(reorder_matrix, e)
            transform = tf.matmul(v, tf.transpose(reorder_matrix))
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
        freqs: dict,
        framechanges: dict
    ):
        tot_dim = self.tot_dim
        ones = tf.ones(tot_dim, dtype=tf.complex128)
        FR = tf.linalg.diag(ones)
        for line in freqs.keys():
            freq = freqs[line]
            framechange = framechanges[line]
            qubit = self.couplings[line].connected[0]
            # TODO extend this to multiple qubits
            ann_oper = self.ann_opers[
                self.names.index(qubit)
            ]
            num_oper = tf.constant(
                np.matmul(ann_oper.T.conj(), ann_oper),
                dtype=tf.complex128
            )
            # if self.dressed:
            #     print('applying transform to FR')
            #     num_oper = tf.matmul(
            #         tf.matmul(tf.linalg.adjoint(self.transform), num_oper),
            #         self.transform
            #     )
            # else:
            #     print('leaving FR as is')
            FR = FR * tf.linalg.expm(
                1.0j * num_oper * (freq * t_final + framechange)
            )

        return FR

    def get_qubit_freqs(self):
        # TODO figure how to get the correct dressed frequencies
        pass

    def get_dephasing_channel(self, t_final, amps):
        tot_dim = self.tot_dim
        ones = tf.ones(tot_dim, dtype=tf.complex128)
        Id = tf_utils.tf_super(tf.linalg.diag(ones))
        deph_ch = Id
        for line in amps.keys():
            amp = amps[line]
            qubit = self.couplings[line].connected[0]
            # TODO extend this to multiple qubits
            ann_oper = self.ann_opers[qubit]
            num_oper = tf.constant(
                np.matmul(ann_oper.T.conj(), ann_oper),
                dtype=tf.complex128
            )
            Z = tf_utils.tf_super(
                tf.linalg.expm(
                    1.0j * num_oper * tf.constant(np.pi, dtype=tf.complex128)
                )
            )
            p = t_final * amp * self.dephasing_strength
            print('dephasing stength: ', p)
            if p.numpy() > 1 or p.numpy() < 0:
                raise ValueError('strengh of dephasing channels outside [0,1]')
                print('dephasing stength: ', p)
            deph_ch = deph_ch * ((1-p) * Id + p * Z)
        return deph_ch

    # def simple_dephasing_channel(self, t_final, amps):
    #     amp = amps['d1']
    #     diag = tf.constant([1, 1, 0, 0], dtype=tf.complex128)
    #     Id = tf.linalg.diag(diag)
    #     Id = tf_utils.tf_super(Id)
    #     deph_ch = Id
    #     Z = tf.constant(
    #         np.array(
    #             [[1, 0, 0, 0],
    #              [0, -1, 0, 0],
    #              [0, 0, 0, 0],
    #              [0, 0, 0, 0]]),
    #         dtype=tf.complex128
    #     )
    #     Z = tf_utils.tf_super(Z)
    #     p = t_final * amp * self.dephasing_strength
    #     if p.numpy() > 1 or p.numpy() < 0:
    #         raise ValueError('strengh of dephasing channels outside [0,1]')
    #         print('dephasing stength: ', p)
    #     deph_ch = deph_ch * ((1-p) * Id + p * Z)
    #     return deph_ch
