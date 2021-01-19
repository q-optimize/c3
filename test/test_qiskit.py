"""Module for testing C3 Integration with Qiskit
"""

from c3.qiskit import C3Provider
from qiskit.quantum_info import Statevector
from qiskit import transpile
from test.conftest import get_test_circuit  # noqa
from test.conftest import get_bell_circuit  # noqa
from qiskit.providers import BackendV1 as Backend
from qiskit import execute

import pytest


@pytest.mark.unit
@pytest.mark.qiskit
def test_backends():
    """Test backends() function which returns all available backends"""
    c3_qiskit = C3Provider()
    for backend in c3_qiskit.backends():
        assert isinstance(backend, Backend)


@pytest.mark.unit
@pytest.mark.qiskit
@pytest.mark.parametrize("backend", ["c3_qasm_simulator"])
def test_get_backend(backend):
    """Test get_backend() which returns the backend with matching name

    Parameters
    ----------
    backend : str
        name of the backend that is to be fetched
    """
    c3_qiskit = C3Provider()
    received_backend = c3_qiskit.get_backend(backend)
    assert isinstance(received_backend, Backend)


@pytest.mark.unit
@pytest.mark.qiskit
@pytest.mark.parametrize("backend", ["c3_qasm_simulator"])
def test_transpile(get_test_circuit, backend):  # noqa
    """Test the transpiling using our backends

    Parameters
    ----------
    get_test_circuit : callable
        pytest fixture for a simple quantum circuit
    backend : str
        name of the backend which is to be tested
    """
    c3_qiskit = C3Provider()
    received_backend = c3_qiskit.get_backend(backend)
    trans_qc = transpile(get_test_circuit, received_backend)
    assert Statevector.from_instruction(get_test_circuit).equiv(
        Statevector.from_instruction(trans_qc)
    )


@pytest.mark.integration
@pytest.mark.qiskit
@pytest.mark.slow
@pytest.mark.parametrize("backend", ["c3_qasm_simulator"])
def test_get_result(get_bell_circuit, backend):  # noqa
    """Test the counts from running a Bell Circuit

    Parameters
    ----------
    get_bell_circuit : callable
        pytest fixture for a bell circuit
    backend : str
        name of the backend which is to be tested
    """
    c3_qiskit = C3Provider()
    received_backend = c3_qiskit.get_backend(backend)
    received_backend.set_device_config("test/quickstart.hjson")
    qc = get_bell_circuit
    job_sim = execute(qc, received_backend, shots=10)
    result_sim = job_sim.result()
    assert result_sim.get_counts(qc) == {"00": 4, "11": 4, "01": 1, "10": 1}
