"""Module for testing C3 Integration with Qiskit
"""

import json
from c3.libraries.constants import GATES
from c3.experiment import Experiment
from c3.c3objs import Quantity as Qty
from c3.qiskit import C3Provider
from c3.qiskit.c3_exceptions import C3QiskitError
from c3.qiskit.c3_gates import (
    RX90pGate,
    RX90mGate,
    RXpGate,
    RY90pGate,
    RY90mGate,
    RYpGate,
    RZ90pGate,
    RZ90mGate,
    RZpGate,
    CRXpGate,
    CRGate,
    CR90Gate,
    SetParamsGate,
)
from qiskit.circuit.library import RXGate, RYGate, RZGate, CRXGate, UnitaryGate
from qiskit.quantum_info import Statevector, Operator
from qiskit import transpile
from qiskit.providers import BackendV1 as Backend
from qiskit import QuantumCircuit
import pytest
import numpy as np


@pytest.mark.unit
@pytest.mark.qiskit
def test_backends():
    """Test backends() function which returns all available backends"""
    c3_qiskit = C3Provider()
    for backend in c3_qiskit.backends():
        assert isinstance(backend, Backend)


@pytest.mark.unit
@pytest.mark.qiskit
@pytest.mark.parametrize("backend", ["c3_qasm_physics_simulator"])
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
def test_transpile(get_test_circuit):  # noqa
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
    received_backend = c3_qiskit.get_backend("c3_qasm_physics_simulator")
    trans_qc = transpile(get_test_circuit, received_backend)
    assert Statevector.from_instruction(get_test_circuit).equiv(
        Statevector.from_instruction(trans_qc)
    )


@pytest.mark.integration
@pytest.mark.qiskit
@pytest.mark.slow
def test_qiskit_physics(get_physics_circuit):
    """API test for qiskit physics simulation"""
    c3_qiskit = C3Provider()
    backend = "c3_qasm_physics_simulator"
    physics_backend = c3_qiskit.get_backend(backend)
    physics_backend.set_device_config("test/qiskit.cfg")
    qc = get_physics_circuit
    qc.measure_all()
    job_sim = physics_backend.run(qc)
    expected_pops = np.array([0, 0, 0.5, 0, 0, 0, 0.5, 0])
    received_pops = np.array(list(job_sim.result().data()["state_pops"].values()))
    np.testing.assert_allclose(received_pops, desired=expected_pops, atol=1e-1)
    result_json = json.dumps(
        job_sim.result().to_dict()
    )  # ensure results can be properly serialised
    assert json.loads(result_json)["backend_name"] == backend


@pytest.mark.unit
@pytest.mark.qiskit
@pytest.mark.slow
def test_qiskit_parameter_update(get_physics_circuit):
    """Test for checking parameters are updated by the gate & options interface"""
    c3_qiskit = C3Provider()
    physics_backend = c3_qiskit.get_backend("c3_qasm_physics_simulator")
    physics_backend.set_device_config("test/qiskit.cfg")
    qc = get_physics_circuit  # TODO use a smaller circuit

    # Test that runtime options are correctly assigned
    opt_map = [[["rx90p[0]", "d1", "gaussian", "amp"]]]
    param = [
        Qty(value=0.5, min_val=0.2, max_val=0.6, unit="V"),
    ]
    _ = physics_backend.run(qc, params=param, opt_map=opt_map)
    physics_backend.c3_exp.pmap.set_opt_map(opt_map)
    np.testing.assert_allclose(
        physics_backend.c3_exp.pmap.get_parameter_dict()[
            "rx90p[0]-d1-gaussian-amp"
        ].numpy(),
        param[0].numpy(),
    )

    # Test that custom gate for setting parameters works
    amp = Qty(value=0.8, min_val=0.2, max_val=1, unit="V")
    opt_map = [[["rx90p[0]", "d1", "gaussian", "amp"]]]
    param_gate = SetParamsGate(params=[[amp.asdict()], opt_map])
    qc.append(param_gate, [0])
    _ = physics_backend.run(qc)
    np.testing.assert_allclose(
        physics_backend.c3_exp.pmap.get_parameter_dict()[
            "rx90p[0]-d1-gaussian-amp"
        ].numpy(),
        amp.numpy(),
    )

    # Test that SetParamsGate not at the end raises an error
    qc.append(RX90pGate(), [0])
    with pytest.raises(C3QiskitError):
        _ = physics_backend.run(qc)


@pytest.mark.parametrize(
    ["backend", "config_file"],
    [
        pytest.param("c3_qasm_physics_simulator", "test/qiskit.cfg", id="physics_sim"),
    ],
)
@pytest.mark.unit
@pytest.mark.qiskit
def test_too_many_qubits(backend, config_file):
    """Check that error is raised when circuit has more qubits than device

    Parameters
    ----------
    backend : tuple
        name and device config of the backend to be tested
    """
    c3_qiskit = C3Provider()
    received_backend = c3_qiskit.get_backend(backend)
    received_backend.set_device_config(config_file)
    qc = QuantumCircuit(4, 4)
    with pytest.raises(C3QiskitError):
        received_backend.run(qc, shots=1000)


@pytest.mark.parametrize(
    ["c3_gate", "c3_qubits", "qiskit_gate", "qiskit_qubits"],
    [
        pytest.param(RX90pGate(), [0], RXGate(theta=np.pi / 2.0), [0], id="rx90p"),
        pytest.param(RX90mGate(), [0], RXGate(theta=-np.pi / 2.0), [0], id="rx90m"),
        pytest.param(RXpGate(), [0], RXGate(theta=np.pi), [0], id="rxp"),
        pytest.param(RY90pGate(), [0], RYGate(theta=np.pi / 2.0), [0], id="ry90p"),
        pytest.param(RY90mGate(), [0], RYGate(theta=-np.pi / 2.0), [0], id="ry90m"),
        pytest.param(RYpGate(), [0], RYGate(theta=np.pi), [0], id="ryp"),
        pytest.param(RZ90pGate(), [0], RZGate(phi=np.pi / 2.0), [0], id="rz90p"),
        pytest.param(RZ90mGate(), [0], RZGate(phi=-np.pi / 2.0), [0], id="rz90m"),
        pytest.param(RZpGate(), [0], RZGate(phi=np.pi), [0], id="rzp"),
        # TODO Fix this dummy test for CRXp once matrix is resolved in c3_gates
        pytest.param(
            CRXpGate,
            [0, 1],
            CRXGate(theta=np.pi),
            [0, 1],
            marks=pytest.mark.xfail,
            id="crxp",
        ),
        pytest.param(CRGate(), [0, 1], UnitaryGate(data=GATES["cr"]), [0, 1], id="cr"),
        pytest.param(
            CR90Gate(), [0, 1], UnitaryGate(data=GATES["cr90"]), [0, 1], id="cr90"
        ),
    ],
)
@pytest.mark.unit
@pytest.mark.qiskit
def test_custom_c3_qiskit_gates(c3_gate, c3_qubits, qiskit_gate, qiskit_qubits):
    """Test custom c3 gates for qiskit have correct matrix representations

    Parameters
    ----------
    c3_gate : Gate
        A qiskit Gate object for c3 custom gates
    c3_qubits : List
        List containing the target qubits
    qiskit_gate : Gate
        Gate object for native qiskit gates
    qiskit_qubits : List
        List containing the target qubits
    """
    # TODO configure and test on c3 perfect simulator

    qc_c3 = QuantumCircuit(2, 2)
    qc_qiskit = QuantumCircuit(2, 2)

    qc_c3.append(c3_gate, c3_qubits)
    op_c3 = Operator(qc_c3)
    qc_qiskit.append(qiskit_gate, qiskit_qubits)
    op_qiskit = Operator(qc_qiskit)

    assert op_c3.equiv(op_qiskit)
    np.testing.assert_allclose(
        c3_gate.to_matrix(), desired=GATES[c3_gate.name], atol=1e-3
    )


@pytest.mark.unit
@pytest.mark.qiskit
@pytest.mark.parametrize(
    ["backend", "config_file"],
    [
        pytest.param("c3_qasm_physics_simulator", "test/qiskit.cfg", id="physics_sim"),
    ],
)
def test_user_provided_c3_exp(backend, config_file):
    """Test for checking user provided C3 Experiment object is correctly assigned"""
    test_exp = Experiment()
    test_exp.load_quick_setup(config_file)
    c3_qiskit = C3Provider()
    received_backend = c3_qiskit.get_backend(backend)
    received_backend.set_c3_experiment(test_exp)
    assert received_backend.c3_exp is test_exp


@pytest.mark.unit
@pytest.mark.qiskit
@pytest.mark.parametrize(
    ["backend"],
    [
        pytest.param("c3_qasm_physics_simulator", id="physics_sim"),
    ],
)
def test_experiment_not_initialised(backend, get_test_circuit):
    """Test for checking error is raised if c3 experiment object is not correctly initialised"""
    c3_qiskit = C3Provider()
    received_backend = c3_qiskit.get_backend(backend)
    qc = get_test_circuit
    with pytest.raises(C3QiskitError):
        received_backend.run(qc)


@pytest.mark.qiskit
@pytest.mark.unit
def test_initial_statevector(get_physics_circuit):
    """Check initial statevector is correctly validated

    Parameters
    ----------
    get_physics_circuit : QuantumCircuit
        Circuit fixture with RX90p and CR90 gates
    """
    c3_qiskit = C3Provider()
    physics_backend = c3_qiskit.get_backend("c3_qasm_physics_simulator")
    physics_backend.set_device_config("test/qiskit.cfg")
    qc = get_physics_circuit
    qc.measure_all()
    with pytest.raises(C3QiskitError):
        # incorrect dimension
        physics_backend.run(qc, initial_statevector=[0, 0, 0, 0, 0, 0, 1])

    with pytest.raises(C3QiskitError):
        # unnormalised
        physics_backend.run(qc, initial_statevector=[1, 1, 1, 1, 1, 1, 1, 1])
