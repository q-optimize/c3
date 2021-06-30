"""
Experiment class that models and simulates the whole experiment.

It combines the information about the model of the quantum device, the control stack
and the operations that can be done on the device.

Given this information an experiment run is simulated, returning either processes,
states or populations.
"""

import os
import copy
import pickle
import itertools
import hjson
import numpy as np
import tensorflow as tf
from typing import Dict
import time

from c3.c3objs import hjson_encode, hjson_decode
from c3.generator.generator import Generator
from c3.parametermap import ParameterMap
from c3.signal.gates import Instruction
from c3.model import Model
from c3.utils.tf_utils import (
    tf_matmul_left,
    tf_state_to_dm,
    tf_super,
    tf_vec_to_dm,
)

from c3.libraries.propagation import (
    tf_batch_propagate,
    tf_propagation_lind,
)
from c3.utils.qt_utils import perfect_single_q_parametric_gate


class Experiment:
    """
    It models all of the behaviour of the physical experiment, serving as a
    host for the individual parts making up the experiment.

    Parameters
    ----------
    pmap: ParameterMap
        including
        model: Model
            The underlying physical device.
        generator: Generator
            The infrastructure for generating and sending control signals to the
            device.
        gateset: GateSet
            A gate level description of the operations implemented by control
            pulses.

    """

    def __init__(self, pmap: ParameterMap = None):
        self.pmap = pmap
        self.opt_gates = None
        self.propagators: Dict[str, tf.Tensor] = {}
        self.partial_propagators: dict = {}
        self.created_by = None
        self.logdir: str = None
        self.propagate_batch_size = None
        self.use_control_fields = True
        self.overwrite_propagators = True  # Keep only currently computed propagators
        self.compute_propagators_timestamp = 0
        self.stop_partial_propagator_gradient = True
        self.evaluate = self.evaluate_legacy

    def enable_qasm(self) -> None:
        """
        Switch the sequencing format to QASM. Will become the default.
        """
        self.evaluate = self.evaluate_qasm

    def set_created_by(self, config):
        """
        Store the config file location used to created this experiment.
        """

        self.created_by = config

    def load_quick_setup(self, filepath: str) -> None:
        """
        Load a quick setup file.

        Parameters
        ----------
        filepath : str
            Location of the configuration file

        """
        with open(filepath, "r") as cfg_file:
            cfg = hjson.loads(cfg_file.read(), object_pairs_hook=hjson_decode)
        self.quick_setup(cfg)

    def quick_setup(self, cfg) -> None:
        """
        Load a quick setup cfg and create all necessary components.

        Parameters
        ----------
        cfg : Dict
            Configuration options

        """
        model = Model()
        model.read_config(cfg["model"])
        gen = Generator()
        gen.read_config(cfg["generator"])

        single_gate_time = cfg["single_qubit_gate_time"]
        v2hz = cfg["v2hz"]
        instructions = []
        sideband = cfg.pop("sideband", None)
        for gate_name, props in cfg["single_qubit_gates"].items():
            target_qubit = model.subsystems[props["qubits"]]
            instr = Instruction(
                name=props["name"],
                targets=[model.names.index(props["qubits"])],
                t_start=0.0,
                t_end=single_gate_time,
                channels=[target_qubit.drive_line],
            )
            instr.quick_setup(
                target_qubit.drive_line,
                target_qubit.params["freq"].get_value() / 2 / np.pi,
                single_gate_time,
                v2hz,
                sideband,
            )
            instructions.append(instr)

        for gate_name, props in cfg["two_qubit_gates"].items():
            qubit_1 = model.subsystems[props["qubit_1"]]
            qubit_2 = model.subsystems[props["qubit_2"]]
            instr = Instruction(
                name=gate_name,
                targets=[
                    model.names.index(props["qubit_1"]),
                    model.names.index(props["qubit_2"]),
                ],
                t_start=0.0,
                t_end=props["gate_time"],
                channels=[qubit_1.drive_line, qubit_2.drive_line],
            )
            instr.quick_setup(
                qubit_1.drive_line,
                qubit_1.params["freq"].get_value() / 2 / np.pi,
                props["gate_time"],
                v2hz,
                sideband,
            )
            instr.quick_setup(
                qubit_2.drive_line,
                qubit_2.params["freq"].get_value() / 2 / np.pi,
                props["gate_time"],
                v2hz,
                sideband,
            )
            instructions.append(instr)

        self.pmap = ParameterMap(instructions, generator=gen, model=model)

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
        self.from_dict(cfg)

    def from_dict(self, cfg: dict) -> None:
        """
        Load experiment from dictionary
        """
        model = Model()
        model.fromdict(cfg["model"])
        generator = Generator()
        generator.fromdict(cfg["generator"])
        pmap = ParameterMap(model=model, generator=generator)
        pmap.fromdict(cfg["instructions"])
        if "options" in cfg:
            for k, v in cfg["options"].items():
                self.__dict__[k] = v
        self.pmap = pmap

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
        exp_dict: Dict[str, dict] = {}
        exp_dict["instructions"] = {}
        for name, instr in self.pmap.instructions.items():
            exp_dict["instructions"][name] = instr.asdict()
        exp_dict["model"] = self.pmap.model.asdict()
        exp_dict["generator"] = self.pmap.generator.asdict()
        exp_dict["options"] = {
            "propagate_batch_size": self.propagate_batch_size,
            "use_control_fields": self.use_control_fields,
            "overwrite_propagators": self.overwrite_propagators,
            "stop_partial_propagator_gradient": self.stop_partial_propagator_gradient,
        }
        return exp_dict

    def __str__(self) -> str:
        return hjson.dumps(self.asdict(), default=hjson_encode)

    def evaluate_legacy(self, sequences):
        """
        Compute the population values for a given sequence of operations.

        Parameters
        ----------
        sequences: str list
            A list of control pulses/gates to perform on the device.

        Returns
        -------
        list
            A list of populations

        """
        model = self.pmap.model
        psi_init = model.tasks["init_ground"].initialise(
            model.drift_ham, model.lindbladian
        )
        self.psi_init = psi_init
        populations = []
        for sequence in sequences:
            psi_t = copy.deepcopy(self.psi_init)
            for gate in sequence:
                psi_t = tf.matmul(self.propagators[gate], psi_t)

            pops = self.populations(psi_t, model.lindbladian)
            populations.append(pops)
        return populations

    def evaluate_qasm(self, sequences):
        """
        Compute the population values for a given sequence (in QASM format) of
        operations.

        Parameters
        ----------
        sequences: dict list
            A list of control pulses/gates to perform on the device in QASM format.

        Returns
        -------
        list
            A list of populations

        """
        model = self.pmap.model
        if "init_ground" in model.tasks:
            psi_init = model.tasks["init_ground"].initialise(
                model.drift_ham, model.lindbladian
            )
        else:
            psi_init = model.get_ground_state()
        self.psi_init = psi_init
        populations = []
        for sequence in sequences:
            psi_t = copy.deepcopy(self.psi_init)
            for gate in sequence:
                psi_t = tf.matmul(self.lookup_gate(**gate), psi_t)

            pops = self.populations(psi_t, model.lindbladian)
            populations.append(pops)
        return populations

    def lookup_gate(self, name, qubits, params=None) -> tf.constant:
        """
        Returns a fixed operation or a parametric virtual Z gate. To be extended to
        general parametric gates.
        """
        if name == "VZ":
            gate = tf.constant(self.get_VZ(qubits, params))
        else:
            gate = self.propagators[name + str(qubits)]
        return gate

    def get_VZ(self, target, params):
        """
        Returns the appropriate Z-rotation.
        """
        dims = self.pmap.model.dims
        return perfect_single_q_parametric_gate("Z", target[0], params[0], dims)

    def process(self, populations, labels=None):
        """
        Apply a readout procedure to a population vector. Very specialized
        at the moment.

        Parameters
        ----------
        populations: list
            List of populations from evaluating.

        labels: list
            List of state labels specifying a subspace.

        Returns
        -------
        list
            A list of processed populations.

        """
        model = self.pmap.model
        populations_final = []
        populations_no_rescale = []
        for pops in populations:
            # TODO: Loop over all model.tasks in a general fashion
            # TODO: Selecting states by label in the case of computational space
            if "conf_matrix" in model.tasks:
                pops = model.tasks["conf_matrix"].confuse(pops)
                if labels is not None:
                    pops_select = 0
                    for label in labels:
                        pops_select += pops[model.comp_state_labels.index(label)]
                    pops = pops_select
                else:
                    pops = tf.reshape(pops, [pops.shape[0]])
            else:
                if labels is not None:
                    pops_select = 0
                    for label in labels:
                        try:
                            pops_select += pops[model.state_labels.index(label)]
                        except ValueError:
                            raise Exception(
                                f"C3:ERROR:State {label} not defined. Available are:\n"
                                f"{model.state_labels}"
                            )
                    pops = pops_select
                else:
                    pops = tf.reshape(pops, [pops.shape[0]])
            if "meas_rescale" in model.tasks:
                populations_no_rescale.append(pops)
                pops = model.tasks["meas_rescale"].rescale(pops)
            populations_final.append(pops)
        return populations_final, populations_no_rescale

    def get_perfect_gates(self, gate_keys: list = None) -> Dict[str, np.array]:
        """Return a perfect gateset for the gate_keys.

        Parameters
        ----------
        gate_keys: list
            (Optional) List of gates to evaluate.

        Returns
        -------
        Dict[str, np.array]
            A dictionary of gate names and np.array representation
            of the corresponding unitary

        Raises
        ------
        Exception
            Raise general exception for undefined gate
        """
        instructions = self.pmap.instructions
        gates = {}
        dims = self.pmap.model.dims
        if gate_keys is None:
            gate_keys = instructions.keys()  # type: ignore
        for gate in gate_keys:
            gates[gate] = instructions[gate].get_ideal_gate(dims)

        # TODO parametric gates

        return gates

    def compute_propagators(self):
        """
        Compute the unitary representation of operations. If no operations are
        specified in self.opt_gates the complete gateset is computed.

        Returns
        -------
        dict
            A dictionary of gate names and their unitary representation.
        """
        model = self.pmap.model
        generator = self.pmap.generator
        instructions = self.pmap.instructions
        gates = {}
        gate_ids = self.opt_gates
        if gate_ids is None:
            gate_ids = instructions.keys()

        for gate in gate_ids:
            try:
                instr = instructions[gate]
            except KeyError:
                raise Exception(
                    f"C3:Error: Gate '{gate}' is not defined."
                    f" Available gates are:\n {list(instructions.keys())}."
                )
            signal = generator.generate_signals(instr)
            U = self.propagation(signal, gate)
            if model.use_FR:
                # TODO change LO freq to at the level of a line
                freqs = {}
                framechanges = {}
                for line, ctrls in instr.comps.items():
                    # TODO calculate properly the average frequency that each qubit sees
                    offset = 0.0
                    for ctrl in ctrls.values():
                        if "freq_offset" in ctrl.params.keys():
                            if ctrl.params["amp"] != 0.0:
                                offset = ctrl.params["freq_offset"].get_value()
                    freqs[line] = tf.cast(
                        ctrls["carrier"].params["freq"].get_value() + offset,
                        tf.complex128,
                    )
                    framechanges[line] = tf.cast(
                        ctrls["carrier"].params["framechange"].get_value(),
                        tf.complex128,
                    )
                t_final = tf.constant(instr.t_end - instr.t_start, dtype=tf.complex128)
                FR = model.get_Frame_Rotation(t_final, freqs, framechanges)
                if model.lindbladian:
                    SFR = tf_super(FR)
                    U = tf.matmul(SFR, U)
                    self.FR = SFR
                else:
                    U = tf.matmul(FR, U)
                    self.FR = FR
            if model.dephasing_strength != 0.0:
                if not model.lindbladian:
                    raise ValueError("Dephasing can only be added when lindblad is on.")
                else:
                    amps = {}
                    for line, ctrls in instr.comps.items():
                        amp, sum = generator.devices["awg"].get_average_amp()
                        amps[line] = tf.cast(amp, tf.complex128)
                    t_final = tf.constant(
                        instr.t_end - instr.t_start, dtype=tf.complex128
                    )
                    dephasing_channel = model.get_dephasing_channel(t_final, amps)
                    U = tf.matmul(dephasing_channel, U)
            gates[gate] = U

        # TODO we might want to move storing of the propagators to the instruction object
        if self.overwrite_propagators:
            self.propagators = gates
        else:
            self.propagators.update(gates)
        self.compute_propagators_timestamp = time.time()
        return gates

    def propagation(self, signal: dict, gate):
        """
        Solve the equation of motion (Lindblad or Schrödinger) for a given control
        signal and Hamiltonians.

        Parameters
        ----------
        signal: dict
            Waveform of the control signal per drive line.
        gate: str
            Identifier for one of the gates.

        Returns
        -------
        unitary
            Matrix representation of the gate.
        """
        model = self.pmap.model

        if self.use_control_fields:
            hamiltonian, hctrls = model.get_Hamiltonians()
            signals = []
            hks = []
            for key in signal:
                signals.append(signal[key]["values"])
                ts = signal[key]["ts"]
                hks.append(hctrls[key])
            signals = tf.cast(signals, tf.complex128)
            hks = tf.cast(hks, tf.complex128)
        else:
            hamiltonian = model.get_Hamiltonian(signal)
            ts_list = [sig["ts"][1:] for sig in signal.values()]
            ts = tf.constant(tf.math.reduce_mean(ts_list, axis=0))
            signals = None
            hks = None
            assert np.all(
                tf.math.reduce_variance(ts_list, axis=0) < 1e-5 * (ts[1] - ts[0])
            )
            assert np.all(
                tf.math.reduce_variance(ts[1:] - ts[:-1]) < 1e-5 * (ts[1] - ts[0])
            )

        # TODO: is this compatible with lindbladian
        if model.max_excitations:
            cutter = model.ex_cutter
            hamiltonian = cutter @ hamiltonian @ cutter.T
            if hks is not None:
                cutter_tf = tf.cast(cutter, tf.complex128)
                hks = tf.matmul(cutter_tf, tf.matmul(hks, cutter_tf, transpose_b=True))

        dt = tf.constant(ts[1].numpy() - ts[0].numpy(), dtype=tf.complex128)

        if model.lindbladian:
            col_ops = model.get_Lindbladians()
            if model.max_excitations:
                cutter = model.ex_cutter
                col_ops = [cutter @ col_op @ cutter.T for col_op in col_ops]
            dUs = tf_propagation_lind(hamiltonian, hks, col_ops, signals, dt)
        else:
            batch_size = (
                self.propagate_batch_size
                if self.propagate_batch_size
                else len(hamiltonian)
            )
            batch_size = (
                len(hamiltonian) if batch_size > len(hamiltonian) else batch_size
            )
            batch_size = tf.constant(batch_size, tf.int32)
            dUs = tf_batch_propagate(
                hamiltonian, hks, signals, dt, batch_size=batch_size
            )

        U = tf_matmul_left(tf.cast(dUs, tf.complex128))
        if model.max_excitations:
            U = cutter.T @ U @ cutter
            ex_cutter = tf.cast(tf.expand_dims(model.ex_cutter, 0), tf.complex128)
            if self.stop_partial_propagator_gradient:
                self.partial_propagators[gate] = tf.stop_gradient(
                    tf.linalg.matmul(
                        tf.linalg.matmul(tf.linalg.matrix_transpose(ex_cutter), dUs),
                        ex_cutter,
                    )
                )
            else:
                self.partial_propagators[gate] = tf.stop_gradient(
                    tf.linalg.matmul(
                        tf.linalg.matmul(tf.linalg.matrix_transpose(ex_cutter), dUs),
                        ex_cutter,
                    )
                )
        else:
            self.partial_propagators[gate] = dUs

        self.ts = ts
        return U

    def set_opt_gates(self, gates):
        """
        Specify a selection of gates to be computed.

        Parameters
        ----------
        gates: Identifiers of the gates of interest. Can contain duplicates.

        """
        if type(gates) is str:
            gates = [gates]
        self.opt_gates = gates

    def set_opt_gates_seq(self, seqs):
        """
        Specify a selection of gates to be computed.

        Parameters
        ----------
        seqs: Identifiers of the sequences of interest. Can contain duplicates.

        """
        self.opt_gates = list(set(itertools.chain.from_iterable(seqs)))

    def set_enable_store_unitaries(self, flag, logdir, exist_ok=False):
        """
        Saving of unitary propagators.

        Parameters
        ----------
        flag: boolean
            Enable or disable saving.
        logdir: str
            File path location for the resulting unitaries.
        """
        self.enable_store_unitaries = flag
        self.logdir = logdir
        if self.enable_store_unitaries:
            os.makedirs(self.logdir + "unitaries/", exist_ok=exist_ok)
            self.store_unitaries_counter = 0

    def store_Udict(self, goal):
        """
        Save unitary as text and pickle.

        Parameter
        ---------
        goal: tf.float64
            Value of the goal function, if used.

        """
        folder = (
            self.logdir
            + "unitaries/eval_"
            + str(self.store_unitaries_counter)
            + "_"
            + str(goal)
            + "/"
        )
        if not os.path.exists(folder):
            os.mkdir(folder)
        with open(folder + "Us.pickle", "wb+") as file:
            pickle.dump(self.propagators, file)
        for key, value in self.propagators.items():
            # Windows is not able to parse ":" as file path
            np.savetxt(folder + key.replace(":", ".") + ".txt", value)

    def populations(self, state, lindbladian):
        """
        Compute populations from a state or density vector.

        Parameters
        ----------
        state: tf.Tensor
            State or densitiy vector.
        lindbladian: boolean
            Specify if conversion to density matrix is needed.

        Returns
        -------
        tf.Tensor
            Vector of populations.
        """
        if lindbladian:
            rho = tf_vec_to_dm(state)
            pops = tf.math.real(tf.linalg.diag_part(rho))
            return tf.reshape(pops, shape=[pops.shape[0], 1])
        else:
            return tf.abs(state) ** 2

    def expect_oper(self, state, lindbladian, oper):
        if lindbladian:
            rho = tf_vec_to_dm(state)
        else:
            rho = tf_state_to_dm(state)
        trace = np.trace(np.matmul(rho, oper))
        return [[np.real(trace)]]  # ,[np.imag(trace)]]
