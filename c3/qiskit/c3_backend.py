import uuid
import time
import numpy as np
import logging

from qiskit import qobj
from qiskit import QuantumCircuit
from qiskit.exceptions import QiskitError
from qiskit.providers import BackendV1 as Backend
from qiskit.providers import Options
from qiskit.providers.models import QasmBackendConfiguration
from qiskit.result import Result
from qiskit.compiler import assemble
from qiskit.qobj.qasm_qobj import QasmQobjExperiment

from .c3_exceptions import C3QiskitError
from .c3_job import C3Job

from typing import Any, Dict
from collections import Counter

logger = logging.getLogger(__name__)


class C3QasmSimulator(Backend):
    """A C3-based Qasm Simulator for Qiskit

    Parameters
    ----------
    Backend : qiskit.providers.BackendV1
        The C3QasmSimulator is derived from BackendV1
    """

    MAX_QUBITS_MEMORY = 10
    _configuration = {
        "backend_name": "c3_qasm_simulator",
        "backend_version": "1.1",
        "n_qubits": min(5, MAX_QUBITS_MEMORY),
        "url": "https://github.com/q-optimize/c3",
        "simulator": True,
        "local": True,
        "conditional": True,
        "open_pulse": False,
        "memory": True,
        "max_shots": 65536,
        "coupling_map": None,
        "description": "A c3 simulator for qasm experiments",
        "basis_gates": ["u1", "u2", "u3", "cx", "id", "unitary"],
        "gates": [
            {
                "name": "u1",
                "parameters": ["lambda"],
                "qasm_def": "gate u1(lambda) q { U(0,0,lambda) q; }",
            },
            {
                "name": "u2",
                "parameters": ["phi", "lambda"],
                "qasm_def": "gate u2(phi,lambda) q { U(pi/2,phi,lambda) q; }",
            },
            {
                "name": "u3",
                "parameters": ["theta", "phi", "lambda"],
                "qasm_def": "gate u3(theta,phi,lambda) q { U(theta,phi,lambda) q; }",
            },
            {
                "name": "cx",
                "parameters": ["c", "t"],
                "qasm_def": "gate cx c,t { CX c,t; }",
            },
            {
                "name": "id",
                "parameters": ["a"],
                "qasm_def": "gate id a { U(0,0,0) a; }",
            },
            {
                "name": "unitary",
                "parameters": ["matrix"],
                "qasm_def": "unitary(matrix) q1, q2,...",
            },
        ],
    }

    DEFAULT_OPTIONS = {"initial_statevector": None, "shots": 1024, "memory": False}
    SHOW_FINAL_STATE = False  # noqa

    def __init__(self, configuration=None, provider=None, **fields):
        super().__init__(
            configuration=(
                configuration or QasmBackendConfiguration.from_dict(self._configuration)
            ),
            provider=provider,
            **fields
        )

    @classmethod
    def _default_options(cls) -> Options:
        return Options(shots=1024, memory=False, initial_statevector=None)

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
        elif isinstance(qobj, qobj.PulseQobj):
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
        # TODO implement interface with C3
        end = time.time()
        # TODO return dict with experiment result
        result = {
            "name": "dummy_name",
            "seed": 2441129,
            "shots": 100,
            "data": {"counts": {"0x9": 5}, "memory": ["0x9", "0xF", "0x1D", "0x9"]},
            "status": "Job Done",
            "success": True,
            "time_taken": 123456,
        }

        return result

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

    def _set_options(self, qobj_config=None, backend_options=None):
        """Set the backend options for all experiments in a qobj"""
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
