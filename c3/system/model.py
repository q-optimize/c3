"""The model class, containing information on the system and its modelling."""
import warnings
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
    max_excitations : int
        Allow only up to max_excitations in the system


    Attributes
    ----------


    """

    def __init__(self, subsystems=None, couplings=None, tasks=None, max_excitations=0):
        self.dressed = True
        self.lindbladian = False
        self.use_FR = True
        self.dephasing_strength = 0.0
        self.params = {}
        self.subsystems: dict = dict()
        self.couplings: dict = dict()
        self.tasks: dict = dict()
        self.drift_ham = None
        self.dressed_drift_ham = None
        self.__hamiltonians = None
        self.__dressed_hamiltonians = None
        if subsystems:
            self.set_components(subsystems, couplings, max_excitations)
        if tasks:
            self.set_tasks(tasks)

    def get_ground_state(self) -> tf.constant:
        gs = [[0] * self.tot_dim]
        gs[0][0] = 1
        return tf.transpose(tf.constant(gs, dtype=tf.complex128))

    def set_components(self, subsystems, couplings=None, max_excitations=0) -> None:
        for comp in subsystems:
            self.subsystems[comp.name] = comp
        for comp in couplings:
            self.couplings[comp.name] = comp
        if len(set(self.couplings.keys()).intersection(self.subsystems.keys())) > 0:
            raise Exception("Do not use same name for multiple devices")
        self.__create_labels()
        self.__create_annihilators()
        self.__create_matrix_representations()
        self.set_max_excitations(max_excitations)

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
        self.tot_dim = int(np.prod(dims))
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

    def set_max_excitations(self, max_excitations) -> None:
        """
        Set the maximum number of excitations in the system used for propagation.
        """
        if max_excitations:
            labels = self.state_labels
            cut_labels = []
            proj = []
            ii = 0
            for li in labels:
                if sum(li) <= max_excitations:
                    cut_labels.append(li)
                    line = [0] * len(labels)
                    line[ii] = 1
                    proj.append(line)
                ii += 1
            # ... If we state labels are changed then final unitaries would have to be changed, too ...
            # self.state_labels = cut_labels
            excitation_cutter = np.array(proj)
            self.ex_cutter = excitation_cutter
        else:
            self.ex_cutter = np.eye(self.tot_dim)
        self.max_excitations = max_excitations

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
        cfg : dict
            configuration file

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
        max_ex = cfg.pop("max_excitations", None)
        self.set_max_excitations(max_ex)

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

    def set_dephasing_strength(self, dephasing_strength):
        self.dephasing_strength = dephasing_strength

    def list_parameters(self):
        ids = []
        for key in self.params:
            ids.append(("Model", key))
        return ids

    def get_Hamiltonians(self):
        if self.dressed:
            return self.dressed_drift_ham, self.dressed_control_hams
        else:
            return self.drift_ham, self.control_hams

    @tf.function
    def get_Hamiltonian(self, signal=None):
        """Get a hamiltonian with an optional signal. This will return an hamiltonian over time.
        Can be used e.g. for tuning the frequency of a transmon, where the control hamiltonian is not easily accessible"""
        if signal is None:
            if self.dressed:
                return self.dressed_drift_ham
            else:
                return self.drift_ham
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
        control_hams = dict()
        hamiltonians = dict()
        for key, sub in self.subsystems.items():
            hamiltonians[key] = sub.get_Hamiltonian()
        for key, line in self.couplings.items():
            hamiltonians[key] = line.get_Hamiltonian()
            if isinstance(line, Drive):
                control_hams[key] = line.get_Hamiltonian(True)

        self.drift_ham = sum(hamiltonians.values())
        self.control_hams = control_hams
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
        e, v = tf.linalg.eigh(self.drift_ham)
        if ordered:
            v_sq = tf.identity(tf.math.real(v * tf.math.conj(v)))

            max_probabilities = tf.expand_dims(tf.reduce_max(v_sq, axis=0), 0)
            if tf.math.reduce_min(max_probabilities) > 0.5:
                reorder_matrix = tf.cast(v_sq > 0.5, tf.float64)
            else:
                failed_states = np.sum(max_probabilities < 0.5)
                min_failed_state = np.argmax(max_probabilities[0] < 0.5)
                warnings.warn(
                    f"""C3 Warning: Some states are overly dressed, trying to recover...{failed_states} states, {min_failed_state} is lowest failed state"""
                )
                vc = v_sq.numpy()
                reorder_matrix = np.zeros_like(vc)
                for i in range(vc.shape[1]):
                    idx = np.unravel_index(np.argmax(vc), vc.shape)
                    vc[idx[0], :] = 0
                    vc[:, idx[1]] = 0
                    reorder_matrix[idx] = 1
                reorder_matrix = tf.constant(reorder_matrix, tf.float64)
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
            reorder_matrix = tf.eye(self.tot_dim)
            eigenframe = tf.math.real(e)
            transform = v

        self.eigenframe = eigenframe
        self.transform = tf.cast(transform, dtype=tf.complex128)
        self.reorder_matrix = reorder_matrix

    def update_dressed(self, ordered=True):
        """Compute the Hamiltonians in the dressed basis by diagonalizing the drift and applying the resulting
        transformation to the control Hamiltonians."""
        self.update_drift_eigen(ordered=ordered)
        dressed_control_hams = {}
        dressed_col_ops = []
        dressed_hamiltonians = dict()
        for k, h in self.__hamiltonians.items():
            dressed_hamiltonians[k] = tf.matmul(
                tf.matmul(tf.linalg.adjoint(self.transform), h), self.transform
            )
        dressed_drift_ham = tf.matmul(
            tf.matmul(tf.linalg.adjoint(self.transform), self.drift_ham), self.transform
        )
        for key in self.control_hams:
            dressed_control_hams[key] = tf.matmul(
                tf.matmul(tf.linalg.adjoint(self.transform), self.control_hams[key]),
                self.transform,
            )
        self.dressed_drift_ham = dressed_drift_ham
        self.dressed_control_hams = dressed_control_hams
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
        es = tf.math.real(tf.linalg.diag_part(self.dressed_drift_ham))
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
            if p.numpy() > 1 or p.numpy() < 0:
                raise ValueError(
                    "Dephasing channel strength {strength} is outside [0,1] range".format(
                        strength=p
                    )
                )
            # TODO: check that this is right (or do you put the Zs together?)
            deph_ch = deph_ch * ((1 - p) * Id + p * Z)
        return deph_ch
