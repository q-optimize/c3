from typing import Any, Dict
from qiskit import QuantumCircuit
import pytest


@pytest.fixture()
def get_test_circuit() -> QuantumCircuit:
    """fixture for sample Quantum Circuit

    Returns
    -------
    QuantumCircuit
        A circuit with a Hadamard, a C-X
    """
    qc = QuantumCircuit(2, 2)
    qc.h(0)
    qc.cx(0, 1)
    return qc


@pytest.fixture()
def get_bell_circuit() -> QuantumCircuit:
    """fixture for Quantum Circuit to make Bell
    State |11> + |00>

    Returns
    -------
    QuantumCircuit
        A circuit with a Hadamard, a C-X and 2 Measures
    """
    qc = QuantumCircuit(2, 2)
    qc.h(0)
    qc.cx(0, 1)
    qc.measure([0, 1], [0, 1])
    return qc


@pytest.fixture()
def get_6_qubit_circuit() -> QuantumCircuit:
    """fixture for 6 qubit Quantum Circuit

    Returns
    -------
    QuantumCircuit
        A circuit with an X on qubit 1
    """
    qc = QuantumCircuit(6, 6)
    qc.x(0)
    qc.cx(0, 1)
    qc.measure([0], [0])
    return qc


@pytest.fixture()
def get_result_qiskit() -> Dict[str, Any]:
    """Fixture for returning sample experiment result

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

    """
    # Result of physics based sim for applying X on qubit 0 in 6 qubits
    counts = {"000000": 164, "010000": 799, "100000": 14}
    return counts
