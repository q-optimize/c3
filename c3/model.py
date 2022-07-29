"""The model class, containing information on the system and its modelling."""
import warnings
import numpy as np
import hjson
import itertools
import copy
import tensorflow as tf
import c3.utils.tf_utils as tf_utils
import c3.utils.qt_utils as qt_utils
from c3.c3objs import hjson_encode, hjson_decode
from c3.libraries.chip import device_lib, Drive, Coupling
from c3.libraries.tasks import task_lib
from typing import Dict, List, Tuple, Union


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
        self.couplings: Dict[str, Union[Drive, Coupling]] = {}
        self.tasks: dict = dict()
        self.drift_ham = None
        self.dressed_drift_ham = None
        self.__hamiltonians = None
        self.__dressed_hamiltonians = None
        self.init_state = None
        if subsystems:
            self.set_components(subsystems, couplings, max_excitations)
        if tasks:
            self.set_tasks(tasks)
        self.controllability = True

    def set_init_state(self, state):
        if self.lindbladian and state.shape[0] != state.shape[1]:
            if state.shape[0] == self.tot_dim:
                self.init_state = tf_utils.tf_state_to_dm(state)
            elif self.lindbladian and state.shape[0] == self.tot_dim**2:
                self.init_state = tf_utils.tf_vec_to_dm(state)
        else:
            self.init_state = state

    def get_ground_state(self) -> tf.constant:
        gs = [[0] * self.tot_dim]
        gs[0][0] = 1
        return tf.transpose(tf.constant(gs, dtype=tf.complex128))

    def get_init_state(self) -> tf.Tensor:
        """Get an initial state. If a task to compute a thermal state is set, return that."""
        if self.init_state is None:
            if "init_ground" in self.tasks:
                print("Initial state not specified. Using thermal state as the initial state.")
                print("You can use model.set_init_state() method to set the initial state.")
                psi_init = self.tasks["init_ground"].initialise(
                    self.drift_ham, self.lindbladian
                )
            else:
                print("Initial state not specified. Using ground state as the initial state.")
                print("You can use model.set_init_state() method to set the initial state.")
                psi_init = self.get_ground_state()
                if self.lindbladian:
                    psi_init = tf_utils.tf_state_to_dm(psi_init)
                self.init_state = psi_init
        else:
            psi_init = self.init_state
        return psi_init

    def __check_drive_connect(self, comp):
        for connect in comp.connected:
            try:
                self.subsystems[connect].drive_line = comp.name
            except KeyError:
                raise KeyError(
                    f"Tried to connect {comp.name}"
                    f" to non-existent device {self.subsystems[connect].name}."
                )

    def set_components(self, subsystems, couplings=None, max_excitations=0) -> None:
        for comp in subsystems:
            self.subsystems[comp.name] = comp
        for comp in couplings:
            self.couplings[comp.name] = comp
            # Check that the target of a drive exists and is store the info in the target.
            if isinstance(comp, Drive):
                self.__check_drive_connect(comp)
            if len(set(comp.connected) - set(self.subsystems.keys())) > 0:
                raise Exception("Tried to connect non-existent devices.")

        if len(set(self.couplings.keys()).intersection(self.subsystems.keys())) > 0:
            raise KeyError("Do not use same name for multiple devices")
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
        self.tot_dim = int(np.prod(dims))
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
            proj = []
            ii = 0
            for li in labels:
                if sum(li) <= max_excitations:
                    line = [0] * len(labels)
                    line[ii] = 1
                    proj.append(line)
                ii += 1
            excitation_cutter = np.array(proj)
            self.ex_cutter = tf.convert_to_tensor(
                excitation_cutter, dtype=tf.complex128
            )
        self.max_excitations = max_excitations

    def cut_excitations(self, op):
        cutter = self.ex_cutter
        return cutter @ op @ tf.transpose(cutter)

    def blowup_excitations(self, op):
        cutter = self.ex_cutter
        return tf.transpose(cutter) @ op @ cutter

    def read_config(self, filepath: str) -> None:
        """
        Load a file and parse it to create a Model object.

        Parameters
        ----------
        filepath : str
            Location of the configuration file

        """
        with open(filepath, "r") as cfg_file:
            cfg = hjson.loads(cfg_file.read(), object_pairs_hook=hjson_decode)
        self.fromdict(cfg)

    def fromdict(self, cfg: dict) -> None:
        """
        Load a file and parse it to create a Model object.

        Parameters
        ----------
        cfg : dict
            configuration file

        """
        subsystems = []
        for name, props in cfg["Qubits"].items():
            props.update({"name": name})
            dev_type = props.pop("c3type")
            subsystems.append(device_lib[dev_type](**props))

        couplings = []
        for name, props in cfg["Couplings"].items():
            props.update({"name": name})
            dev_type = props.pop("c3type")
            this_dev = device_lib[dev_type](**props)
            couplings.append(this_dev)

        if "Tasks" in cfg:
            tasks = []
            for name, props in cfg["Tasks"].items():
                props.update({"name": name})
                task_type = props.pop("c3type")
                task = task_lib[task_type](**props)
                tasks.append(task)
            self.set_tasks(tasks)

        if "use_dressed_basis" in cfg:
            self.dressed = cfg["use_dressed_basis"]
        self.set_components(subsystems, couplings)
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
            hjson.dump(self.asdict(), cfg_file, default=hjson_encode)

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
        tasks = {}
        for name, task in self.tasks.items():
            tasks[name] = task.asdict()
        return {
            "Qubits": qubits,
            "Couplings": couplings,
            "Tasks": tasks,
            "max_excitations": self.max_excitations,
        }

    def __str__(self) -> str:
        return hjson.dumps(self.asdict(), default=hjson_encode)

    def set_dressed(self, dressed):
        """
        Go to a dressed frame where static couplings have been eliminated.

        Parameters
        ----------
        dressed : boolean

        """
        self.dressed = dressed
        self.update_model()

    def set_lindbladian(self, lindbladian: bool) -> None:
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
        drift = []
        controls = []

        if self.dressed:
            drift = self.dressed_drift_ham
            controls = self.dressed_control_hams
        else:
            drift = self.drift_ham
            controls = self.control_hams
        if self.max_excitations:
            drift = self.cut_excitations(drift)
            controls = self.cut_excitations(controls)
        return drift, controls

    def get_sparse_Hamiltonians(self):
        drift, controls = self.get_Hamiltonians
        sparse_drift = self.blowup_excitations(drift)
        sparse_controls = tf.vectorized_map(self.blowup_excitations, controls)
        return sparse_drift, sparse_controls

    def get_Hamiltonian(self, signal=None):
        """Get a hamiltonian with an optional signal. This will return an hamiltonian over time.
        Can be used e.g. for tuning the frequency of a transmon, where the control hamiltonian is not easily accessible.
        If max.excitation is non-zero the resulting Hamiltonian is cut accordingly"""
        if signal is None:
            if self.dressed:
                signal_hamiltonian = self.dressed_drift_ham
            else:
                signal_hamiltonian = self.drift_ham
        else:
            if self.dressed:
                hamiltonians = copy.deepcopy(self.__dressed_hamiltonians)
                transform = self.transform
            else:
                hamiltonians = copy.deepcopy(self.__hamiltonians)
                transform = None
            for key, sig in signal.items():
                if key in self.subsystems:
                    hamiltonians[key] = self.subsystems[key].get_Hamiltonian(
                        sig, transform
                    )
                elif key in self.couplings:
                    hamiltonians[key] = self.couplings[key].get_Hamiltonian(
                        sig, transform
                    )
                else:
                    raise Exception(f"Signal channel {key} not in model systems")

            signal_hamiltonian = sum(
                [
                    tf.expand_dims(h, 0) if len(h.shape) == 2 else h
                    for h in hamiltonians.values()
                ]
            )

        if self.max_excitations:
            signal_hamiltonian = self.cut_excitations(signal_hamiltonian)

        return signal_hamiltonian

    def get_sparse_Hamiltonian(self, signal=None):
        return self.blowup_excitations(self.get_Hamiltonian(signal))

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
            if isinstance(line, Coupling):
                hamiltonians[key] = line.get_Hamiltonian()
            if isinstance(line, Drive):
                control_hams[key] = line.get_Hamiltonian(signal=True)

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
            if line in self.couplings:
                qubit = self.couplings[line].connected[0]
            elif line in self.subsystems:
                qubit = line
            else:
                raise Exception(
                    f"Component {line} not found in couplings or subsystems"
                )
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

    def Hs_of_t(self, signal, interpolate_res=2):
        """
        Generate a list of Hamiltonians for each time step of interpolated signal for Runge-Kutta Methods.
        
        Args:
            signal (_type_): Input signal
            interpolate_res (int, optional): Interpolation resolution according to RK method. Defaults to 2.
            L_dag_L (tf.tensor, optional): List of {L^\dagger L} where L represents the collapse operators.
                                           Defaults to None. This is only used for stochastic case.

        Returns:
            dict: List of Hamiltonians (or effective Hamiltonians for stochastic case) for each time step.
        """
        h0, hctrls = self.get_Hamiltonians()

        ts_list = []
        signals = []
        hks = []
        for key in signal:
            ts_list.append(signal[key]["ts"])
            signals.append(signal[key]["values"])
            hks.append(hctrls[key])

        ts = tf.math.reduce_mean(ts_list, axis=0)
        # Only do the safety check outside of graph mode for performance reasons.
        # When using graph mode, the safety check will still be executed ONCE during tracing
        if tf.executing_eagerly() and not tf.reduce_all(
            tf.math.reduce_variance(ts_list, axis=0) < (1e-5 * (ts[1] - ts[0]))
        ):
            raise Exception("C3Error:Something with the times happend.")
        if tf.executing_eagerly() and not tf.reduce_all(
            tf.math.reduce_variance(ts[1:] - ts[:-1]) < 1e-5 * (ts[1] - ts[0])  # type: ignore
        ):
            raise Exception("C3Error:Something with the times happend.")
        dt = ts[1] - ts[0]
        dt = tf.cast(dt, dtype=tf.complex128)
        

        signals_interp = []
        for sig in signals:
            sig_new = tf_utils.interpolateSignal(ts, sig, interpolate_res)
            signals_interp.append(sig_new)

        cflds = tf.cast(signals_interp, tf.complex128)
        hks = tf.cast(hks, tf.complex128)

        Hs = self.calculate_sum_Hs(h0, hks, cflds)
        ts = tf.cast(ts, dtype=tf.complex128)

        return {"Hs": Hs, "ts": ts, "dt": dt}

    def calculate_sum_Hs(self, h0, hks, cflds):
        control_field = tf.reshape(
            tf.transpose(cflds), 
            (tf.shape(cflds)[1], tf.shape(cflds)[0], 1, 1)
        )
        hk = tf.multiply(control_field, hks)
        Hs = tf.reduce_sum(hk, axis=1)
        return Hs + h0
