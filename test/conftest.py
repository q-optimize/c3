from qiskit import QuantumCircuit
import pytest


@pytest.fixture()
def get_test_circuit() -> QuantumCircuit:
    """fixture for sample Quantum Circuit

    Returns
    -------
    QuantumCircuit
        A circuit with a Hadamard, a C-X and 2 Measures
    """
    qc = QuantumCircuit(2, 2)
    qc.h(0)
    qc.cx(0, 1)
    return qc
