import numpy as np
import tensorflow as tf
from typing import Any, Dict, Iterator, Tuple
from qiskit import QuantumCircuit, ClassicalRegister, QuantumRegister
from c3.utils.tf_utils import (
    tf_super,
    tf_choi_to_chi,
    tf_abs,
    super_to_choi,
    tf_project_to_comp,
)
from c3.parametermap import ParameterMap
from c3.generator.generator import Generator
from c3.generator.devices import Crosstalk
from c3.libraries.constants import Id, X, Y, Z
from c3.c3objs import Quantity
import pytest


@pytest.fixture()
def get_exp_problem() -> Iterator[Tuple[np.ndarray, np.ndarray]]:
    theta = 2 * np.pi * np.random.rand(1)
    """Testing that exponentiation methods are almost equal in the numpy sense.
    Check that, given P = a*X+b*Y+c*Z with a, b, c random normalized numbers,
    exp(i theta P) = cos(theta)*Id + sin(theta)*P"""
    a = np.random.rand(1)
    b = np.random.rand(1)
    c = np.random.rand(1)
    norm = np.sqrt(a ** 2 + b ** 2 + c ** 2)
    # Normalized random coefficients
    a = a / norm
    b = b / norm
    c = c / norm

    P = a * X + b * Y + c * Z
    theta = 2 * np.pi * np.random.rand(1)
    rot = theta * P
    res = np.cos(theta) * Id + 1j * np.sin(theta) * P
    yield rot, res


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
def get_bad_circuit() -> QuantumCircuit:
    """fixture for Quantum Circuit with
    unsupported operations

    Returns
    -------
    QuantumCircuit
        A circuit with a Conditional
    """
    q = QuantumRegister(1)
    c = ClassicalRegister(1)
    qc = QuantumCircuit(q, c)
    qc.x(q[0]).c_if(c, 0)
    qc.measure(q, c)
    return qc


@pytest.fixture()
def get_3_qubit_circuit() -> QuantumCircuit:
    """fixture for 3 qubit Quantum Circuit

    Returns
    -------
    QuantumCircuit
    """
    qc = QuantumCircuit(3, 3)
    qc.rx(np.pi / 2, 0)
    qc.rx(np.pi / 2, 1)

    qc.measure(
        [
            0,
            1,
            2,
        ],
        [
            0,
            1,
            2,
        ],
    )
    return qc


@pytest.fixture()
def get_result_qiskit() -> Dict[str, Dict[str, Any]]:
    """Fixture for returning sample experiment result

    Returns
    -------
    Dict[str, Dict[str, Any]]
            A dictionary of results for physics simulation and perfect gates
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
    perfect_counts = {"000": 250, "010": 250, "100": 250, "110": 250}

    counts_dict = {
        "c3_qasm_perfect_simulator": perfect_counts,
    }
    return counts_dict


@pytest.fixture
def get_error_process():
    """Fixture for a constant unitary

    Returns
    -------
    np.array
        Unitary on a large Hilbert space that needs to be projected down correctly and
        compared to an ideal representation in the computational space.
    """
    U_actual = (
        1
        / np.sqrt(2)
        * np.array(
            [
                [1, 0, 0, -1.0j, 0, 0, 0, 0, 0],
                [0, 1, 0, 0, -1.0j, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0],
                [-1.0j, 0, 0, 1, 0, 0, 0, 0, 0],
                [0, -1.0j, 0, 0, 1, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 45, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0],
            ]
        )
    )

    lvls = [3, 3]
    U_ideal = (
        1
        / np.sqrt(2)
        * np.array(
            [[1, 0, -1.0j, 0], [0, 1, 0, -1.0j], [-1.0j, 0, 1, 0], [0, -1.0j, 0, 1]]
        )
    )
    Lambda = tf.matmul(
        tf.linalg.adjoint(tf_project_to_comp(U_actual, lvls, to_super=False)), U_ideal
    )
    return Lambda


@pytest.fixture
def get_average_fidelitiy(get_error_process):
    lvls = [3, 3]
    Lambda = get_error_process
    d = 4
    err = tf_super(Lambda)
    choi = super_to_choi(err)
    chi = tf_choi_to_chi(choi, dims=lvls)
    fid = tf_abs((chi[0, 0] / d + 1) / (d + 1))
    return fid


@pytest.fixture()
def get_xtalk_pmap() -> ParameterMap:
    xtalk = Crosstalk(
        name="crosstalk",
        channels=["TC1", "TC2"],
        crosstalk_matrix=Quantity(
            value=[[1, 0], [0, 1]],
            min_val=[[0, 0], [0, 0]],
            max_val=[[1, 1], [1, 1]],
            unit="",
        ),
    )

    gen = Generator(devices={"crosstalk": xtalk})
    pmap = ParameterMap(generator=gen)
    pmap.set_opt_map([[["crosstalk", "crosstalk_matrix"]]])
    return pmap


@pytest.fixture()
def get_test_signal() -> Dict:
    return {
        "TC1": {"values": tf.linspace(0, 100, 101)},
        "TC2": {"values": tf.linspace(100, 200, 101)},
    }


@pytest.fixture()
def get_test_dimensions() -> list:
    """Functions in qt_utils that allow arbitrary numbers of dimensions will be tested for all dimensions in this
    list."""
    return [3, 5, 10, 50]
