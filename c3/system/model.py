"""The model class, containing information on the system and its modelling."""

import numpy as np
import hjson
import itertools
import tensorflow as tf
import c3.utils.tf_utils as tf_utils
import c3.utils.qt_utils as qt_utils
from c3.system.chip import device_lib, Coupling, Drive


class Model:
    """
    What the theorist thinks about from the system.

    Class to store information about our system/problem/device. Different
    models can represent the same system.

    Parameters
    ----------
    subsystems : list
        List of individual, non-interacting physical components like qubits or resonators
    couplings : list
        List of interaction operators between subsystems, like couplings or drives.
    tasks : list
        Badly named list of processing steps like line distortions and read out modelling


    Attributes
    ----------
    H0: :class: Drift Hamiltonian


    """

    def __init__(self, subsystems=None, couplings=None, tasks=None):
        self.dressed = False
        self.lindbladian = False
        self.use_FR = True
        self.dephasing_strength = 0.0
        self.params = {}
        self.subsystems = {}
        self.couplings = {}
        self.tasks = {}
        if subsystems:
            self.set_components(subsystems, couplings)
        if tasks:
            self.set_tasks(tasks)

    def set_components(self, subsystems, couplings=None) -> None:
        for comp in subsystems:
            self.subsystems[comp.name] = comp
        for comp in couplings:
            self.couplings[comp.name] = comp
        self.__create_labels()
        self.__create_annihilators()
        self.__create_matrix_representations()

    def set_tasks(self, tasks) -> None:
        for task in tasks:
            self.tasks[task.name] = task

    def __create_labels(self) -> None:
        """
        Iterate over the physical subsystems and create labeling for the product space.
        """
        dims = []
        names = []
        state_labels = []
        comp_state_labels = []
        for subs in self.subsystems.values():
            dims.append(subs.hilbert_dim)
            names.append(subs.name)
            # TODO user defined labels
            state_labels.append(list(range(subs.hilbert_dim)))
            comp_state_labels.append([0, 1])
        self.tot_dim = np.prod(dims)
        self.names = names
        self.dims = dims
        self.state_labels = list(itertools.product(*state_labels))
        self.comp_state_labels = list(itertools.product(*comp_state_labels))

    def __create_annihilators(self) -> None:
        """
        Construct the annihilation operators for the full system via Kronecker product.
        """
        ann_opers = []
        dims = self.dims
        for indx in range(len(dims)):
            a = np.diag(np.sqrt(np.arange(1, dims[indx])), k=1)
            ann_opers.append(
                qt_utils.hilbert_space_kron(a, indx, dims)
            )
        self.ann_opers = ann_opers

    def __create_matrix_representations(self) -> None:
        """
        Using the annihilation operators as basis, compute the matrix represenations.
        """
        indx = 0
        ann_opers = self.ann_opers
        for subs in self.subsystems.values():
            subs.init_Hs(ann_opers[indx])
            subs.init_Ls(ann_opers[indx])
            subs.set_subspace_index(indx)
            indx += 1
        for line in self.couplings.values():
            conn = line.connected
            opers_list = []
            for sub in conn:
                try:
                    indx = self.names.index(sub)
                except ValueError as ve:
                    raise Exception(f"C3:ERROR: Trying to couple to unkown subcomponent: {sub}")
                opers_list.append(self.ann_opers[indx])
            line.init_Hs(opers_list)
        self.update_model()

    def read_config(self, filepath: str) -> None:
        """
        Load a file and parse it to create a Model object.

        Parameters
        ----------
        filepath : str
            Location of the configuration file

        """
        with open(filepath, "r") as cfg_file:
            cfg = hjson.loads(cfg_file.read())
        for name, props in cfg["Qubits"].items():
            props.update({"name": name})
            dev_type = props.pop("c3type")
            self.subsystems[name] = device_lib[dev_type](**props)
        for name, props in cfg["Couplings"].items():
            props.update({"name": name})
            dev_type = props.pop("c3type")
            self.couplings[name] = device_lib[dev_type](**props)
        self.__create_labels()
        self.__create_annihilators()
        self.__create_matrix_representations()

    def write_config(self, filepath: str) -> None:
        """
        Write dictionary to a HJSON file.
        """
        with open(filepath, "w") as cfg_file:
            hjson.dump(self.asdict(), cfg_file)

    def asdict(self) -> dict:
        """
        Return a dictionary compatible with config files.
        """
        qubits = {}
        for name, qubit in self.subsystems.items():
            qubits[name] = qubit.asdict()
        couplings = {}
        for name, coup in self.couplings.items():
            couplings[name] = coup.asdict()
        return {"Qubits": qubits, "Couplings": couplings}

    def __str__(self) -> str:
        return hjson.dumps(self.asdict())

    def set_dressed(self, dressed):
        """
        Go to a dressed frame where static couplings have been eliminated.

        Parameters
        ----------
        dressed : boolean

        """
        self.dressed = dressed
        self.update_model()

    def set_lindbladian(self, lindbladian):
        """
        Set whether to include open system dynamics.

        Parameters
        ----------
        lindbladian : boolean


        """
        self.lindbladian = lindbladian
        self.update_model()

    def set_FR(self, use_FR):
        """
        Setter for the frame rotation option for adjusting the individual rotating frames of
        qubits when using gate sequences
        """
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
        """Recompute the matrix representations of the Hamiltonians."""
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
        """Compute the eigendecomposition of the drift Hamiltonian and store both the
        Eigenenergies and the transformation matrix."""
        # TODO Raise error if dressing unsuccesful
        e, v = tf.linalg.eigh(self.drift_H)
        reorder_matrix = tf.cast(tf.round(tf.math.real(v)), tf.complex128)
        if ordered:
            eigenframe = tf.linalg.matvec(reorder_matrix, e)
            transform = tf.matmul(v, tf.transpose(reorder_matrix))
        else:
            eigenframe = tf.linalg.diag(e)
            transform = v
        self.eigenframe = eigenframe
        self.transform = transform

    def update_dressed(self):
        """Compute the Hamiltonians in the dressed basis by diagonalizing the drift and applying
        the resulting transformation to the control Hamiltonians."""
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
        """
        Compute the frame rotation needed to align Lab frame and rotating Eigenframes of the qubits.

        Parameters
        ----------
        t_final : tf.float64
            Gate length
        freqs : list
            Frequencies of the local oscillators.
        framechanges : list
            List of framechanges. A phase shift applied to the control signal to compensate
            relative phases of drive oscillator and qubit.

        Returns
        -------
        tf.Tensor
            A (diagonal) propagator that adjust phases
        """
        exponent = tf.constant(0., dtype=tf.complex128)
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
            # TODO test this
            if self.dressed:
                num_oper = tf.matmul(
                    tf.matmul(tf.linalg.adjoint(self.transform), num_oper),
                    self.transform
                )
            exponent = exponent +\
                1.0j * num_oper * (freq * t_final + framechange)
        FR = tf.linalg.expm(exponent)
        return FR

    def get_qubit_freqs(self):
        # TODO figure how to get the correct dressed frequencies
        pass

    def get_dephasing_channel(self, t_final, amps):
        """
        Compute the matrix of the dephasing channel to be applied on the operation.

        Parameters
        ----------
        t_final : tf.float64
            Duration of the operation.
        amps : dict of tf.float64
            Dictionary of average amplitude on each drive line.
        Returns
        -------
        tf.tensor
            Matrix representation of the dephasing channel.
        """

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
