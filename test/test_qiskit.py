"""Module for testing C3 Integration with Qiskit
"""

import json
from c3.qiskit import C3Provider
from c3.qiskit.c3_exceptions import C3QiskitError
from qiskit.quantum_info import Statevector
from qiskit import transpile
from qiskit.providers import BackendV1 as Backend
from qiskit import execute, QuantumCircuit
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
@pytest.mark.parametrize("backend", ["c3_qasm_perfect_simulator"])
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
@pytest.mark.parametrize("backend", ["c3_qasm_perfect_simulator"])
def test_transpile(get_test_circuit, backend):  # noqa
    """Test the transpiling using our backends.
    Should throw error due to missing H gate

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
@pytest.mark.parametrize("backend", ["c3_qasm_perfect_simulator"])
def test_get_result(get_6_qubit_circuit, backend, get_result_qiskit):  # noqa
    """Test the counts from running a 6 qubit Circuit

    Parameters
    ----------
    get_6_qubit_circuit : callable
        pytest fixture for a 6 qubit circuit
    backend : str
        name of the backend which is to be tested
    simulation_type: str
        physics based or perfect gates simulation
    get_result_qiskit: callable
        pytest fixture for experiment results
    """
    c3_qiskit = C3Provider()
    received_backend = c3_qiskit.get_backend(backend)
    received_backend.set_device_config("test/quickstart.hjson")
    received_backend.disable_flip_labels()
    qc = get_6_qubit_circuit
    job_sim = execute(qc, received_backend, shots=1000)
    result_sim = job_sim.result()

    # TODO: Test results with qiskit style qubit indexing
    # qiskit_simulator = Aer.get_backend("qasm_simulator")
    # qiskit_counts = execute(qc, qiskit_simulator, shots=1000).result().get_counts(qc)
    # assert result_sim.get_counts(qc) == qiskit_counts

    # Test results with original C3 style qubit indexing
    received_backend.disable_flip_labels()
    no_flip_counts = get_result_qiskit[backend]
    job_sim = execute(qc, received_backend, shots=1000)
    result_sim = job_sim.result()
    result_json = json.dumps(
        result_sim.to_dict()
    )  # ensure results can be properly serialised
    assert json.loads(result_json)["backend_name"] == backend
    assert result_sim.get_counts() == no_flip_counts


@pytest.mark.unit
@pytest.mark.qiskit
@pytest.mark.parametrize("backend", ["c3_qasm_perfect_simulator"])
def test_get_exception(get_bad_circuit, backend):  # noqa
    """Test to check exceptions

    Parameters
    ----------
    get_bad_circuit : callable
        pytest fixture for a circuit with conditional operation
    backend : str
        name of the backend which is to be tested
    """
    c3_qiskit = C3Provider()
    received_backend = c3_qiskit.get_backend(backend)
    received_backend.set_device_config("test/quickstart.hjson")
    qc = get_bad_circuit

    with pytest.raises(C3QiskitError):
        execute(qc, received_backend, shots=1000)


def test_qiskit_physics():
    c3_qiskit = C3Provider()
    physics_backend = c3_qiskit.get_backend("c3_qasm_physics_simulator")
    physics_backend.set_device_config("test/qiskit.cfg")
    qc = QuantumCircuit(2, 2)
    qc.x(0)
    qc.cx(0, 1)
    job_sim = execute(qc, physics_backend)
    print(job_sim.result().get_counts())
