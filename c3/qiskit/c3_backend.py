import uuid
import time
import numpy as np
import logging
import warnings

from qiskit import qobj
from qiskit import QuantumCircuit
from qiskit.exceptions import QiskitError
from qiskit.providers import BackendV1 as Backend
from qiskit.providers import Options
from qiskit.providers.models import QasmBackendConfiguration
from qiskit.result import Result
from qiskit.compiler import assemble
from qiskit.qobj.qasm_qobj import QasmQobjExperiment
from qiskit.qobj.pulse_qobj import PulseQobj

from c3.experiment import Experiment

from .c3_exceptions import C3QiskitError
from .c3_job import C3Job
from .c3_backend_utils import get_init_ground_state, get_sequence, flip_labels

from typing import Any, Dict, List
from abc import ABC, abstractclassmethod, abstractmethod

logger = logging.getLogger(__name__)


class C3QasmSimulator(Backend, ABC):
    """An Abtract Base Class for C3 Qasm Simulators for Qiskit.
    This class CAN NOT be instantiated directly.
    Classes derived from this must compulsorily implement ::

        def __init__(self, configuration=None, provider=None, **fields):

        def _default_options(cls) -> None:

        def run_experiment(self, experiment: QasmQobjExperiment) -> Dict[str, Any]:

    Parameters
    ----------
    Backend : qiskit.providers.BackendV1
        The C3QasmSimulator is derived from BackendV1
    ABC : abc.ABC
        Helper class for defining Abstract classes using ABCMeta
    """

    @abstractclassmethod
    def _default_options(cls) -> None:
        raise NotImplementedError("This must be implemented in the derived class")

    def set_device_config(self, config_file: str) -> None:
        """Set the path to the config for the device

        Parameters
        ----------
        config_file : str
            path to hjson file storing the configuration
            for all device parameters for simulation
        """
        self._device_config = config_file
        self.c3_exp.load_quick_setup(self._device_config)

    def set_c3_experiment(self, exp: Experiment) -> None:
        """Set user-provided c3 experiment object for backend

        Parameters
        ----------
        exp : Experiment
            C3 experiment object
        """
        self.c3_exp = exp

    def get_labels(self, format: str = "qiskit") -> List[str]:
        """Return state labels for the system

        Parameters
        ----------
        format : str, optional
            How to format the state labels, by default "qiskit"

        Returns
        -------
        List[str]
            A list of state labels in hex if qiskit format
            and decimal if c3 format ::

                labels = ['0x1', ...]

                labels = ['01', '02', ...]

        Raises
        ------
        C3QiskitError
            When an supported format is passed
        """
        if format == "qiskit":
            labels = [
                hex(i)
                for i in range(
                    0,
                    pow(
                        self._number_of_levels,
                        self._number_of_qubits,
                    ),
                )
            ]
        elif format == "c3":
            labels = [
                "".join(str(label)) for label in self.c3_exp.pmap.model.state_labels
            ]
        else:
            raise C3QiskitError("Incorrect format specifier for get_labels")
        return labels

    def disable_flip_labels(self) -> None:
        """Disable flipping of labels
        State Labels are flipped before returning results
        to match Qiskit style qubit indexing convention
        This function allows disabling of the flip
        """
        self._flip_labels = False

    def locate_measurements(self, instructions_list: List[Dict]) -> List[int]:
        """Locate the indices of measurement operations in circuit

        Parameters
        ----------
        instructions_list : List[Dict]
            Instructions List in Qasm style

        Returns
        -------
        List[int]
            The indices where measurement operations occur
        """
        meas_index: List[int] = []
        meas_index = [
            index
            for index, instruction in enumerate(instructions_list)
            if instruction["name"] == "measure"
        ]
        return meas_index

    def sanitize_instructions_list(self, instructions_list: List[Dict]) -> List[Dict]:
        """Sanitize instructions list by removing unsupported operations

        Parameters
        ----------
        instructions_list : List[Dict]
            Qasm style list of instructions represented by dicts

        Returns
        -------
        List[Dict]
            Sanitized instruction list

        Raises
        -------
        UserWarning
            Warns user about unsupported operations in circuit
        """
        sanitized_instructions = [
            instruction
            for instruction in instructions_list
            if instruction["name"] not in self.UNSUPPORTED_OPERATIONS
        ]
        if sanitized_instructions != instructions_list:
            warnings.warn(
                f"The following operations are not supported yet: {self.UNSUPPORTED_OPERATIONS}"
            )
        return sanitized_instructions

    def generate_shot_readout(self):
        """Generate shot style readout from population

        Returns
        -------
        List[int]
            List of shots for each output state
        """
        # TODO a sophisticated readout/measurement routine (w/ SPAM)
        return (np.round(self.pops_array * self._shots)).astype("int32").tolist()

    def run(self, qobj: qobj.Qobj, **backend_options) -> C3Job:
        """Parse and run a Qobj

        Parameters
        ----------
        qobj : Qobj
            The Qobj payload for the experiment
        backend_options : dict
            backend options

        Returns
        -------
        C3Job
            An instance of the C3Job (derived from JobV1) with the result

        Raises
        ------
        QiskitError
            Support for Pulse Jobs is not implemented

        Notes
        -----
        backend_options: Is a dict of options for the backend. It may contain
                * "initial_statevector": vector_like

        The "initial_statevector" option specifies a custom initial statevector
        for the simulator to be used instead of the all zero state. This size of
        this vector must be correct for the number of qubits in all experiments
        in the qobj.

        Example::

            backend_options = {
                "initial_statevector": np.array([1, 0, 0, 1j]) / np.sqrt(2),
            }
        """

        if isinstance(qobj, (QuantumCircuit, list)):
            qobj = assemble(qobj, self, **backend_options)
            qobj_options = qobj.config
        elif isinstance(qobj, PulseQobj):
            raise QiskitError("Pulse jobs are not accepted")
        else:
            qobj_options = qobj.config
        self._set_options(qobj_config=qobj_options, backend_options=backend_options)
        job_id = str(uuid.uuid4())
        job = C3Job(self, job_id, self._run_job(job_id, qobj))
        return job

    def _run_job(self, job_id, qobj):
        """Run experiments in qobj

        Parameters
        ----------
        job_id : str
            unique id for the job
        qobj : Qobj
            job description

        Returns
        -------
        Result
            Result object
        """
        self._validate(qobj)
        result_list = []
        self._shots = qobj.config.shots
        self._memory = getattr(qobj.config, "memory", False)
        self._qobj_config = qobj.config
        if not hasattr(self.c3_exp.pmap, "model"):
            raise C3QiskitError(
                "Experiment Object has not been correctly initialised. \nUse set_device_config() or set_c3_experiment()"
            )
        start = time.time()
        for experiment in qobj.experiments:
            result_list.append(self.run_experiment(experiment))
        end = time.time()
        result = {
            "backend_name": self.name(),
            "backend_version": self._configuration.backend_version,
            "qobj_id": qobj.qobj_id,
            "job_id": job_id,
            "results": result_list,
            "status": "COMPLETED",
            "success": True,
            "time_taken": (end - start),
            "header": qobj.header.to_dict(),
        }

        return Result.from_dict(result)

    @abstractmethod
    def run_experiment(self, experiment: QasmQobjExperiment) -> Dict[str, Any]:
        raise NotImplementedError("This must be implemented in the derived class")

    def _validate(self, qobj):
        """Semantic validations of the qobj which cannot be done via schemas."""
        n_qubits = qobj.config.n_qubits
        max_qubits = self.configuration().n_qubits
        if n_qubits > max_qubits:
            raise C3QiskitError(
                "Number of qubits {} ".format(n_qubits)
                + "is greater than maximum ({}) ".format(max_qubits)
                + 'for "{}".'.format(self.name())
            )
        for experiment in qobj.experiments:
            name = experiment.header.name
            if experiment.config.memory_slots == 0:
                logger.warning(
                    'No classical registers in circuit "%s", ' "counts will be empty.",
                    name,
                )
            elif "measure" not in [op.name for op in experiment.instructions]:
                logger.warning(
                    'No measurements in circuit "%s", '
                    "classical register will remain all zeros.",
                    name,
                )

    def _validate_initial_statevector(self):
        """Raise an error when experiment tries to set initial statevector

        Raises
        ------
        C3QiskitError
            Error for statevector initialisation not implemented
        """
        if self._initial_statevector is not None:
            raise C3QiskitError(
                "Setting initial statevector is not implemented in this simulator"
            )
        else:
            pass

    def _initialize_statevector(self):
        """Raise an error when experiment tries to set initial statevector

        Raises
        ------
        C3QiskitError
            Error for statevector initialisation not implemented
        """
        if self._initial_statevector is not None:
            raise C3QiskitError(
                "Setting initial statevector is not implemented in this simulator"
            )
        else:
            pass

    def _set_options(self, qobj_config=None, backend_options=None):
        """Qiskit stock method to Set the backend options for all experiments in a qobj"""
        # Reset default options
        self._initial_statevector = self.options.get("initial_statevector")
        if "backend_options" in backend_options and backend_options["backend_options"]:
            backend_options = backend_options["backend_options"]

        # Check for custom initial statevector in backend_options first,
        # then config second
        if "initial_statevector" in backend_options:
            self._initial_statevector = np.array(
                backend_options["initial_statevector"], dtype=complex
            )
        elif hasattr(qobj_config, "initial_statevector"):
            self._initial_statevector = np.array(
                qobj_config.initial_statevector, dtype=complex
            )
        if self._initial_statevector is not None:
            # Check the initial statevector is normalized
            norm = np.linalg.norm(self._initial_statevector)
            if round(norm, 12) != 1:
                raise C3QiskitError(
                    "initial statevector is not normalized: "
                    + "norm {} != 1".format(norm)
                )


class C3QasmPerfectSimulator(C3QasmSimulator):
    """A C3-based perfect gates simulator for Qiskit

    Parameters
    ----------
    C3QasmSimulator : c3.qiskit.c3_backend.C3QasmSimulator
        Inherits the C3QasmSimulator and implements a perfect gate simulator
    """

    # TODO List correct set of basis gates

    MAX_QUBITS_MEMORY = 20
    _configuration = {
        "backend_name": "c3_qasm_perfect_simulator",
        "backend_version": "0.1",
        "n_qubits": MAX_QUBITS_MEMORY,
        "url": "https://github.com/q-optimize/c3",
        "simulator": True,
        "local": True,
        "conditional": False,
        "open_pulse": False,
        "memory": False,
        "max_shots": 65536,
        "coupling_map": None,
        "description": "A c3 simulator for qasm experiments with perfect gates",
        "basis_gates": [
            "cx",
            "cz",
            "iSwap",
            "id",
            "x",
            "y",
            "z",
            "rx",
            "ry",
            "rz",
            "rzx",
        ],
        "gates": [],
    }

    DEFAULT_OPTIONS = {"initial_statevector": None, "shots": 1024, "memory": False}

    def __init__(self, configuration=None, provider=None, **fields):
        super().__init__(
            configuration=(
                configuration or QasmBackendConfiguration.from_dict(self._configuration)
            ),
            provider=provider,
            **fields,
        )
        # Define attributes in __init__.
        self._local_random = np.random.RandomState()
        self._classical_memory = 0
        self._classical_register = 0
        self._statevector = 0
        self._number_of_cmembits = 0
        self._number_of_qubits = 0
        self._shots = 0
        self._memory = False
        self._initial_statevector = self.options.get("initial_statevector")
        self._qobj_config = None
        # TEMP
        self._sample_measure = False
        self._flip_labels = True
        self.c3_exp = Experiment()

    @classmethod
    def _default_options(cls) -> Options:
        return Options(shots=1024, memory=False, initial_statevector=None)

    def run_experiment(self, experiment: QasmQobjExperiment) -> Dict[str, Any]:
        """Run an experiment (circuit) and return a single experiment result

        Parameters
        ----------
        experiment : QasmQobjExperiment
            experiment from qobj experiments list

        Returns
        -------
        Dict[str, Any]
            A result dictionary which looks something like::

            {
            "name": name of this experiment (obtained from qobj.experiment header)
            "seed": random seed used for simulation
            "shots": number of shots used in the simulation
            "data":
                {
                "counts": {'0x9': 5, ...},
                "memory": ['0x9', '0xF', '0x1D', ..., '0x9']
                },
            "status": status string for the simulation
            "success": boolean
            "time_taken": simulation time of this single experiment
            }

        Raises
        ------
        C3QiskitError
            If an error occured
        """
        start = time.time()

        # setup C3 Experiment
        exp = self.c3_exp
        pmap = exp.pmap
        instructions = pmap.instructions

        # initialise parameters
        self._number_of_qubits = len(pmap.model.subsystems)
        if self._number_of_qubits < experiment.config.n_qubits:
            raise C3QiskitError("Not enough qubits on device to run circuit")

        # TODO (Check) Assume all qubits have same Hilbert dims
        self._number_of_levels = pmap.model.dims[0]

        # Validate the dimension of initial statevector if set
        self._validate_initial_statevector()

        # TODO set simulator seed, check qiskit python qasm simulator
        # qiskit-terra/qiskit/providers/basicaer/qasm_simulator.py
        seed_simulator = 2441129

        # convert qasm instruction set to c3 sequence
        sequence = get_sequence(experiment.instructions)

        # unique operations
        gate_keys = list(set(sequence))

        # validate gates
        for gate in gate_keys:
            if gate not in instructions.keys():
                raise C3QiskitError(
                    "Gate {gate} not found in Device Instruction Set: {instructions}".format(
                        gate=gate, instructions=list(instructions.keys())
                    )
                )

        perfect_gates = exp.get_perfect_gates(gate_keys)

        # initialise state
        psi_init = get_init_ground_state(self._number_of_qubits, self._number_of_levels)
        psi_t = psi_init.numpy()
        pop_t = exp.populations(psi_t, False)

        # compute final state
        for gate in sequence:
            psi_t = np.matmul(perfect_gates[gate], psi_t)
            pops = exp.populations(psi_t, False)
            pop_t = np.append(pop_t, pops, axis=1)

        # generate shots style readout
        self.pops_array = pop_t.T[-1]
        shots_data = self.generate_shot_readout()

        # generate state labels
        output_labels = self.get_labels()

        # create results dict
        counts = dict(zip(output_labels, shots_data))

        # keep only non-zero states
        counts = dict(filter(lambda elem: elem[1] != 0, counts.items()))

        # flipping state labels to match qiskit style qubit indexing convention
        # default is to flip labels to qiskit style, use disable_flip_labels()
        if self._flip_labels:
            counts = flip_labels(counts)

        end = time.time()

        exp_result = {
            "name": experiment.header.name,
            "header": experiment.header.to_dict(),
            "shots": self._shots,
            "seed": seed_simulator,
            "status": "DONE",
            "success": True,
            "data": {"counts": counts},
            "time_taken": (end - start),
        }

        return exp_result


class C3QasmPhysicsSimulator(C3QasmSimulator):
    """A C3-based perfect gates simulator for Qiskit

    Parameters
    ----------
    C3QasmSimulator : c3.qiskit.c3_backend.C3QasmSimulator
        Inherits the C3QasmSimulator and implements a physics based simulator
    """

    MAX_QUBITS_MEMORY = 10
    _configuration = {
        "backend_name": "c3_qasm_physics_simulator",
        "backend_version": "0.1",
        "n_qubits": MAX_QUBITS_MEMORY,
        "url": "https://github.com/q-optimize/c3",
        "simulator": True,
        "local": True,
        "conditional": False,
        "open_pulse": False,
        "memory": False,
        "max_shots": 65536,
        "coupling_map": None,  # TODO Coupling map from config file
        "description": "A physics based c3 simulator for qasm experiments",
        "basis_gates": [  # TODO Basis gates from config file
            "cx",
            "rx",
        ],
        "gates": [],
    }

    DEFAULT_OPTIONS = {"initial_statevector": None, "shots": 1024, "memory": False}

    def __init__(self, configuration=None, provider=None, **fields):
        super().__init__(
            configuration=(
                configuration or QasmBackendConfiguration.from_dict(self._configuration)
            ),
            provider=provider,
            **fields,
        )
        # Define attributes in __init__.
        self._local_random = np.random.RandomState()
        self._classical_memory = 0
        self._classical_register = 0
        self._statevector = 0
        self._number_of_cmembits = 0
        self._number_of_qubits = 0
        self._shots = 0
        self._memory = False
        self._initial_statevector = self.options.get("initial_statevector")
        self._qobj_config = None
        self._sample_measure = False
        self.UNSUPPORTED_OPERATIONS = ["measure", "barrier"]
        self.c3_exp = Experiment()

    @classmethod
    def _default_options(cls) -> Options:
        return Options(shots=1024, memory=False, initial_statevector=None)

    def run_experiment(self, experiment: QasmQobjExperiment) -> Dict[str, Any]:
        """Run an experiment (circuit) and return a single experiment result

        Parameters
        ----------
        experiment : QasmQobjExperiment
            experiment from qobj experiments list

        Returns
        -------
        Dict[str, Any]
            A result dictionary which looks something like::

            {
            "name": name of this experiment (obtained from qobj.experiment header)
            "seed": random seed used for simulation
            "shots": number of shots used in the simulation
            "data":
                {
                "counts": {'0x9: 5, ...},
                "memory": ['0x9', '0xF', '0x1D', ..., '0x9']
                },
            "status": status string for the simulation
            "success": boolean
            "time_taken": simulation time of this single experiment
            }

        Raises
        ------
        C3QiskitError
            If an error occured
        """
        start = time.time()

        # setup C3 Experiment
        exp = self.c3_exp
        exp.enable_qasm()
        exp.compute_propagators()  # TODO only simulate qubits used in circuit
        pmap = exp.pmap

        # initialise parameters
        self._number_of_qubits = len(pmap.model.subsystems)
        if self._number_of_qubits < experiment.config.n_qubits:
            raise C3QiskitError("Not enough qubits on device to run circuit")

        # TODO (Check) Assume all qubits have same Hilbert dims
        self._number_of_levels = pmap.model.dims[0]

        # Validate the dimension of initial statevector if set
        self._validate_initial_statevector()

        # TODO set simulator seed, check qiskit python qasm simulator
        # qiskit-terra/qiskit/providers/basicaer/qasm_simulator.py
        seed_simulator = 2441129
        instructions_list = [
            instruction.to_dict() for instruction in experiment.instructions
        ]
        sanitized_instructions = self.sanitize_instructions_list(instructions_list)

        pops = exp.evaluate([sanitized_instructions])
        pop1s, _ = exp.process(pops)

        # C3 stores labels in exp.pmap.model.state_labels
        meas_index = self.locate_measurements(instructions_list)
        self.pops_array = pop1s[0].numpy()
        if meas_index:
            counts_data = self.generate_shot_readout()
        else:
            counts_data = self.pops_array.tolist()
        counts = dict(zip(self.get_labels(format="qiskit"), counts_data))
        state_pops = dict(zip(self.get_labels(format="c3"), self.pops_array.tolist()))

        # flipping state labels to match qiskit style qubit indexing convention
        # default is to flip labels to qiskit style, use disable_flip_labels()
        # if self._flip_labels:
        #     counts = flip_labels(counts)

        end = time.time()

        exp_result = {
            "name": experiment.header.name,
            "header": experiment.header.to_dict(),
            "shots": self._shots,
            "seed": seed_simulator,
            "status": "DONE",
            "success": True,
            "data": {
                "counts": counts,
                "state_pops": state_pops,
            },
            "time_taken": (end - start),
        }

        return exp_result
