"""The model class, containing information on the system and its modelling."""

import numpy as np
import hjson
import itertools
import copy
import tensorflow as tf
import c3.utils.tf_utils as tf_utils
import c3.utils.qt_utils as qt_utils
from c3.system.chip import device_lib, Drive
from typing import List, Tuple


class Model:
    """
    What the theorist thinks about from the system.

    Class to store information about our system/problem/device. Different
    models can represent the same system.

    Parameters
    ----------
    subsystems : list
        List of individual, non-interacting physical components like qubits or
        resonators
    couplings : list
        List of interaction operators between subsystems, like couplings or drives.
    tasks : list
        Badly named list of processing steps like line distortions and read out
        modeling


    Attributes
    ----------
    H0: :class: Drift Hamiltonian


    """

    def __init__(self, subsystems=None, couplings=None, tasks=None):
        self.dressed = True
        self.lindbladian = False
        self.use_FR = True
        self.dephasing_strength = 0.0
        self.params = {}
        self.subsystems: dict = dict()
        self.couplings: dict = dict()
        self.tasks: dict = dict()
        self.cut_excitations = 0
        self.drift_H = None
        self.dressed_drift_H = None
        self.__hamiltonians = None
        self.__dressed_hamiltonians = None
        if subsystems:
            self.set_components(subsystems, couplings)
        if tasks:
            self.set_tasks(tasks)

    def set_components(self, subsystems, couplings=None) -> None:
        for comp in subsystems:
            self.subsystems[comp.name] = comp
        for comp in couplings:
            self.couplings[comp.name] = comp
        if len(set(self.couplings.keys()).intersection(self.subsystems.keys())) > 0:
            raise Exception("Do not use same name for multiple devices")
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
            ann_opers.append(qt_utils.hilbert_space_kron(a, indx, dims))
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
                    raise Exception(
                        f"C3:ERROR: Trying to couple to unkown subcomponent: {sub}"
                    ) from ve
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
        self.fromdict(cfg)

    def fromdict(self, cfg: dict) -> None:
        """
        Load a file and parse it to create a Model object.

        Parameters
        ----------
        filepath : str
            Location of the configuration file

        """
        for name, props in cfg["Qubits"].items():
            props.update({"name": name})
            dev_type = props.pop("c3type")
            self.subsystems[name] = device_lib[dev_type](**props)
        for name, props in cfg["Couplings"].items():
            props.update({"name": name})
            dev_type = props.pop("c3type")
            self.couplings[name] = device_lib[dev_type](**props)
            if dev_type == "Drive":
                for connection in self.couplings[name].connected:
                    try:
                        self.subsystems[connection].drive_line = name
                    except KeyError as ke:
                        raise Exception(
                            f"C3:ERROR: Trying to connect drive {name} to unkown "
                            f"target {connection}."
                        ) from ke
        if "use_dressed_basis" in cfg:
            self.dressed = cfg["use_dressed_basis"]
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
        Setter for the frame rotation option for adjusting the individual rotating
        frames of qubits when using gate sequences
        """
        self.use_FR = use_FR

    def set_cut_excitations(self, n_cut):
        """
        Set if the outputed hamiltonians should be cut to states only with summed excitations up to n_cut
        Parameters
        ----------
        n_cut: number of maximum excitations in system. 0 corresponds to no additional cutting of the hilbert space.

        Returns
        -------

        """
        self.cut_excitations = n_cut

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

    @tf.function
    def get_Hamiltonian(self, signal=None):
        """Get a hamiltonian with an optional signal. This will return an hamiltonian over time.
        Can be used e.g. for tuning the frequency of a transmon, where the control hamiltonian is not easily accessible"""
        if signal is None:
            if self.dressed:
                return self.dressed_drift_H
            else:
                return self.drift_H
        if self.dressed:
            hamiltonians = copy.deepcopy(self.__dressed_hamiltonians)
            transform = self.transform
        else:
            hamiltonians = copy.deepcopy(self.__hamiltonians)
            transform = None
        for key, sig in signal.items():
            if key in self.subsystems:
                hamiltonians[key] = self.subsystems[key].get_Hamiltonian(sig, transform)
            elif key in self.couplings:
                hamiltonians[key] = self.couplings[key].get_Hamiltonian(sig, transform)
            else:
                raise Exception(f"Signal channel {key} not in model systems")
        signal_hamiltonian = sum(
            [
                tf.expand_dims(h, 0) if len(h.shape) == 2 else h
                for h in hamiltonians.values()
            ]
        )
        return signal_hamiltonian

    def get_reduced_indices(self, num_inner_dims=2):
        if self.cut_excitations > 0:
            idxs = np.argwhere(
                np.array([sum(s) for s in self.state_labels]) <= self.cut_excitations
            )
        else:
            return None, self.tot_dim
        out_shape = [len(idxs)] * num_inner_dims + [num_inner_dims]
        reduced_indices = tf.constant(
            np.reshape(list(itertools.product(idxs, repeat=num_inner_dims)), out_shape)
        )
        return reduced_indices, self.tot_dim

    def get_Lindbladians(self):
        if self.dressed:
            return self.dressed_col_ops
        else:
            return self.col_ops

    def update_model(self, ordered=True):
        self.update_Hamiltonians()
        if self.lindbladian:
            self.update_Lindbladians()
        if self.dressed:
            self.update_dressed(ordered=ordered)

    def update_Hamiltonians(self):
        """Recompute the matrix representations of the Hamiltonians."""
        control_Hs = dict()
        hamiltonians = dict()
        for key, sub in self.subsystems.items():
            hamiltonians[key] = sub.get_Hamiltonian()
        for key, line in self.couplings.items():
            hamiltonians[key] = line.get_Hamiltonian()
            if isinstance(line, Drive):
                control_Hs[key] = line.get_Hamiltonian(True)

        self.drift_H = sum(hamiltonians.values())
        self.control_Hs = control_Hs
        self.__hamiltonians = hamiltonians

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
        if ordered:
            reorder_matrix = tf.cast(
                (
                    tf.expand_dims(tf.reduce_max(tf.abs(v) ** 2, axis=1), 1)
                    == tf.abs(v) ** 2
                ),
                tf.float64,
            )
            signed_rm = tf.cast(
                # TODO determine if the changing of sign is needed
                # (by looking at TC_eneregies_bases I see no difference)
                # reorder_matrix, dtype=tf.complex128
                tf.sign(tf.math.real(v)) * reorder_matrix,
                dtype=tf.complex128,
            )
            eigenframe = tf.linalg.matvec(reorder_matrix, tf.math.real(e))
            transform = tf.matmul(v, tf.transpose(signed_rm))
        else:
            eigenframe = tf.math.real(e)
            transform = v
        self.eigenframe = eigenframe
        self.transform = tf.cast(transform, dtype=tf.complex128)

    def update_dressed(self, ordered=True):
        """Compute the Hamiltonians in the dressed basis by diagonalizing the drift and applying the resulting
        transformation to the control Hamiltonians."""
        self.update_drift_eigen(ordered=ordered)
        dressed_col_ops = []
        dressed_hamiltonians = dict()
        for k, h in self.__hamiltonians.items():
            dressed_hamiltonians[k] = tf.matmul(
                tf.matmul(tf.linalg.adjoint(self.transform), h), self.transform
            )
        dressed_drift_H = sum(dressed_hamiltonians.values())
        self.dressed_drift_H = dressed_drift_H
        self.__dressed_hamiltonians = dressed_hamiltonians
        if self.lindbladian:
            for col_op in self.col_ops:
                dressed_col_ops.append(
                    tf.matmul(
                        tf.matmul(tf.linalg.adjoint(self.transform), col_op),
                        self.transform,
                    )
                )
            self.dressed_col_ops = dressed_col_ops

    def get_Frame_Rotation(self, t_final: np.float64, freqs: dict, framechanges: dict):
        """
        Compute the frame rotation needed to align Lab frame and rotating Eigenframes
        of the qubits.

        Parameters
        ----------
        t_final : tf.float64
            Gate length
        freqs : list
            Frequencies of the local oscillators.
        framechanges : list
            List of framechanges. A phase shift applied to the control signal to
            compensate relative phases of drive oscillator and qubit.

        Returns
        -------
        tf.Tensor
            A (diagonal) propagator that adjust phases
        """
        exponent = tf.constant(0.0, dtype=tf.complex128)
        for line in freqs.keys():
            freq = freqs[line]
            framechange = framechanges[line]
            qubit = self.couplings[line].connected[0]
            # TODO extend this to multiple qubits
            ann_oper = self.ann_opers[self.names.index(qubit)]
            num_oper = tf.constant(
                np.matmul(ann_oper.T.conj(), ann_oper), dtype=tf.complex128
            )
            # TODO test dressing of FR
            exponent = exponent + 1.0j * num_oper * (freq * t_final + framechange)
        if len(exponent.shape) == 0:
            return tf.eye(self.tot_dim, dtype=tf.complex128)
        FR = tf.linalg.expm(exponent)
        return FR

    def get_qubit_freqs(self) -> List[float]:
        # TODO figure how to get the correct dressed frequencies
        pass
        es = tf.math.real(tf.linalg.diag_part(self.dressed_drift_H))
        frequencies = []
        for i in range(len(self.dims)):
            state = [0] * len(self.dims)
            state[i] = 1
            idx = self.state_labels.index(tuple(state))
            freq = float(es[idx] - es[0]) / 2 / np.pi
            frequencies.append(freq)
        return frequencies

    def get_state_index(self, state: Tuple) -> int:
        return self.state_labels.index(tuple(state))

    def get_state_indeces(self, states: List[Tuple]) -> List[int]:
        return [self.get_state_index(s) for s in states]

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
            ann_oper = self.ann_opers[self.names.index(qubit)]
            num_oper = tf.constant(
                np.matmul(ann_oper.T.conj(), ann_oper), dtype=tf.complex128
            )
            Z = tf_utils.tf_super(
                tf.linalg.expm(
                    1.0j * num_oper * tf.constant(np.pi, dtype=tf.complex128)
                )
            )
            p = t_final * amp * self.dephasing_strength
            print("dephasing stength: ", p)
            if p.numpy() > 1 or p.numpy() < 0:
                raise ValueError("strengh of dephasing channels outside [0,1]")
                print("dephasing stength: ", p)
            # TODO: check that this is right (or do you put the Zs together?)
            deph_ch = deph_ch * ((1 - p) * Id + p * Z)
        return deph_ch
