import numpy as np
import tensorflow as tf
import copy
import os
import tempfile
from typing import Any, Dict, Iterator, Tuple
from qiskit import QuantumCircuit, ClassicalRegister, QuantumRegister
from c3.utils.tf_utils import (
    tf_super,
    tf_choi_to_chi,
    tf_abs,
    super_to_choi,
    tf_project_to_comp,
)
from c3.experiment import Experiment
from c3.model import Model
from c3.libraries.chip import Qubit, Coupling, Drive
from c3.signal.pulse import Envelope, Carrier
from c3.libraries.envelopes import no_drive, gaussian_nonorm
from c3.libraries.hamiltonians import int_XX, x_drive
from c3.parametermap import ParameterMap
from c3.signal.gates import Instruction
from c3.generator.generator import Generator
from c3.generator.devices import (
    AWG,
    Crosstalk,
    DigitalToAnalog,
    LO,
    Mixer,
    VoltsToHertz,
)
from c3.libraries.constants import Id, X, Y, Z
from c3.c3objs import Quantity
from c3.optimizers.optimalcontrol import OptimalControl
from c3.libraries.fidelities import unitary_infid_set
from c3.libraries.algorithms import algorithms
from c3.qiskit.c3_gates import RX90pGate, CR90Gate
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
    norm = np.sqrt(a**2 + b**2 + c**2)
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
def get_physics_circuit() -> QuantumCircuit:
    """fixture for Quantum Circuit for physics simulation

    Returns
    -------
    QuantumCircuit
        A circuit with a RX90p, and a CR90
    """
    qc = QuantumCircuit(3)
    qc.append(RX90pGate(), [0])
    qc.append(CR90Gate(), [0, 1])
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


@pytest.fixture()
def get_two_qubit_chip() -> Experiment:
    """Setup a two qubit example with pre-optimized gates."""
    qubit_lvls = 2
    freq_q1 = 5e9
    q1 = Qubit(
        name="Q1",
        desc="Qubit 1",
        freq=Quantity(value=freq_q1, min_val=4.995e9, max_val=5.005e9, unit="Hz 2pi"),
        hilbert_dim=qubit_lvls,
    )

    freq_q2 = 5.6e9
    q2 = Qubit(
        name="Q2",
        desc="Qubit 2",
        freq=Quantity(value=freq_q2, min_val=5.595e9, max_val=5.605e9, unit="Hz 2pi"),
        hilbert_dim=qubit_lvls,
    )

    coupling_strength = 20e6
    q1q2 = Coupling(
        name="Q1-Q2",
        desc="coupling",
        comment="Coupling qubit 1 to qubit 2",
        connected=["Q1", "Q2"],
        strength=Quantity(
            value=coupling_strength, min_val=-1 * 1e3, max_val=200e6, unit="Hz 2pi"
        ),
        hamiltonian_func=int_XX,
    )

    drive = Drive(
        name="d1",
        desc="Drive 1",
        comment="Drive line 1 on qubit 1",
        connected=["Q1"],
        hamiltonian_func=x_drive,
    )
    drive2 = Drive(
        name="d2",
        desc="Drive 2",
        comment="Drive line 2 on qubit 2",
        connected=["Q2"],
        hamiltonian_func=x_drive,
    )

    model = Model(
        subsystems=[q1, q2],  # Individual, self-contained components
        couplings=[q1q2],
        drives=[drive, drive2],  # Interactions between components
    )

    model.set_lindbladian(False)
    model.set_dressed(True)

    sim_res = 100e9  # Resolution for numerical simulation
    awg_res = 2e9  # Realistic, limited resolution of an AWG
    v2hz = 1e9

    generator = Generator(
        devices={
            "LO": LO(name="lo", resolution=sim_res, outputs=1),
            "AWG": AWG(name="awg", resolution=awg_res, outputs=1),
            "DigitalToAnalog": DigitalToAnalog(
                name="dac", resolution=sim_res, inputs=1, outputs=1
            ),
            "Mixer": Mixer(name="mixer", inputs=2, outputs=1),
            "VoltsToHertz": VoltsToHertz(
                name="v_to_hz",
                V_to_Hz=Quantity(
                    value=v2hz, min_val=0.9e9, max_val=1.1e9, unit="Hz 2pi/V"
                ),
                inputs=1,
                outputs=1,
            ),
        },
        chains={
            "d1": {
                "LO": [],
                "AWG": [],
                "DigitalToAnalog": ["AWG"],
                "Mixer": ["LO", "DigitalToAnalog"],
                "VoltsToHertz": ["Mixer"],
            },
            "d2": {
                "LO": [],
                "AWG": [],
                "DigitalToAnalog": ["AWG"],
                "Mixer": ["LO", "DigitalToAnalog"],
                "VoltsToHertz": ["Mixer"],
            },
        },
    )

    t_final = 7e-9  # Time for single qubit gates
    amp = 359e-3  # prev optimized
    sideband = 50e6
    shift = coupling_strength**2 / (freq_q2 - freq_q1)
    gauss_params_single = {
        "amp": Quantity(value=amp, min_val=0.1, max_val=0.6, unit="V"),
        "t_final": Quantity(
            value=t_final, min_val=0.5 * t_final, max_val=1.5 * t_final, unit="s"
        ),
        "sigma": Quantity(
            value=t_final / 4, min_val=t_final / 8, max_val=t_final / 2, unit="s"
        ),
        "xy_angle": Quantity(
            value=0.0, min_val=-0.5 * np.pi, max_val=2.5 * np.pi, unit="rad"
        ),
        "freq_offset": Quantity(
            value=-sideband - shift, min_val=-56 * 1e6, max_val=-48 * 1e6, unit="Hz 2pi"
        ),
    }

    gauss_env_single = Envelope(
        name="gauss",
        desc="Gaussian comp for single-qubit gates",
        params=gauss_params_single,
        shape=gaussian_nonorm,
        use_t_before=True,
    )

    nodrive_env = Envelope(
        name="no_drive",
        params={
            "t_final": Quantity(
                value=t_final, min_val=0.5 * t_final, max_val=1.5 * t_final, unit="s"
            )
        },
        shape=no_drive,
    )

    lo_freq_q1 = 5e9 + sideband
    carrier_parameters = {
        "freq": Quantity(value=lo_freq_q1, min_val=4.5e9, max_val=6e9, unit="Hz 2pi"),
        "framechange": Quantity(
            value=0.0, min_val=-np.pi, max_val=3 * np.pi, unit="rad"
        ),
    }

    carr = Carrier(
        name="carrier",
        desc="Frequency of the local oscillator",
        params=carrier_parameters,
    )

    lo_freq_q2 = 5.6e9 + sideband
    carr_2 = copy.deepcopy(carr)
    carr_2.params["freq"].set_value(lo_freq_q2)

    rx90p_q1 = Instruction(
        name="rx90p",
        targets=[0],
        t_start=0.0,
        t_end=t_final,
        channels=["d1", "d2"],
        params={"use_t_before": True},
    )
    rx90p_q2 = Instruction(
        name="rx90p",
        targets=[1],
        t_start=0.0,
        t_end=t_final,
        channels=["d1", "d2"],
        params={"use_t_before": True},
    )
    QId_q1 = Instruction(
        name="id",
        targets=[0],
        t_start=0.0,
        t_end=t_final,
        channels=["d1", "d2"],
        params={"use_t_before": True},
    )
    QId_q2 = Instruction(
        name="id",
        targets=[1],
        t_start=0.0,
        t_end=t_final,
        channels=["d1", "d2"],
        params={"use_t_before": True},
    )

    rx90p_q1.add_component(gauss_env_single, "d1")
    rx90p_q1.add_component(carr, "d1")
    rx90p_q1.add_component(nodrive_env, "d2")
    rx90p_q1.add_component(copy.deepcopy(carr_2), "d2")
    rx90p_q1.comps["d2"]["carrier"].params["framechange"].set_value(
        (-sideband * t_final) * 2 * np.pi % (2 * np.pi)
    )

    rx90p_q2.add_component(copy.deepcopy(gauss_env_single), "d2")
    rx90p_q2.add_component(carr_2, "d2")
    rx90p_q2.add_component(nodrive_env, "d1")
    rx90p_q2.add_component(copy.deepcopy(carr), "d1")
    rx90p_q2.comps["d1"]["carrier"].params["framechange"].set_value(
        (-sideband * t_final) * 2 * np.pi % (2 * np.pi)
    )

    QId_q1.add_component(nodrive_env, "d1")
    QId_q1.add_component(copy.deepcopy(carr), "d1")
    QId_q1.add_component(nodrive_env, "d2")
    QId_q1.add_component(copy.deepcopy(carr_2), "d2")
    QId_q2.add_component(copy.deepcopy(nodrive_env), "d2")
    QId_q2.add_component(copy.deepcopy(carr_2), "d2")
    QId_q2.add_component(nodrive_env, "d1")
    QId_q2.add_component(copy.deepcopy(carr), "d1")

    Y90p_q1 = copy.deepcopy(rx90p_q1)
    Y90p_q1.name = "ry90p"
    X90m_q1 = copy.deepcopy(rx90p_q1)
    X90m_q1.name = "rx90m"
    Y90m_q1 = copy.deepcopy(rx90p_q1)
    Y90m_q1.name = "ry90m"
    Y90p_q1.comps["d1"]["gauss"].params["xy_angle"].set_value(0.5 * np.pi)
    X90m_q1.comps["d1"]["gauss"].params["xy_angle"].set_value(np.pi)
    Y90m_q1.comps["d1"]["gauss"].params["xy_angle"].set_value(1.5 * np.pi)
    single_q_gates = [QId_q1, rx90p_q1, Y90p_q1, X90m_q1, Y90m_q1]

    Y90p_q2 = copy.deepcopy(rx90p_q2)
    Y90p_q2.name = "ry90p"
    X90m_q2 = copy.deepcopy(rx90p_q2)
    X90m_q2.name = "rx90m"
    Y90m_q2 = copy.deepcopy(rx90p_q2)
    Y90m_q2.name = "ry90m"
    Y90p_q2.comps["d2"]["gauss"].params["xy_angle"].set_value(0.5 * np.pi)
    X90m_q2.comps["d2"]["gauss"].params["xy_angle"].set_value(np.pi)
    Y90m_q2.comps["d2"]["gauss"].params["xy_angle"].set_value(1.5 * np.pi)
    single_q_gates.extend([QId_q2, rx90p_q2, Y90p_q2, X90m_q2, Y90m_q2])

    pmap = ParameterMap(single_q_gates, generator, model)

    pmap.set_opt_map(
        [
            [["rx90p[0]", "d1", "gauss", "amp"]],
            [["rx90p[0]", "d1", "gauss", "freq_offset"]],
            [["rx90p[0]", "d1", "gauss", "xy_angle"]],
        ]
    )
    return Experiment(pmap, sim_res=sim_res)


@pytest.fixture(
    params=[
        "single_eval",
        "tf_sgd",
        "lbfgs",
        "lbfgs_grad_free",
        "cmaes",
        "cma_pre_lbfgs",
    ]
)
def get_OC_optimizer(request, get_two_qubit_chip) -> OptimalControl:
    """Create a general optimizer object with each algorithm with a decorator in the lib."""
    exp = get_two_qubit_chip
    exp.set_opt_gates(["rx90p[0]"])
    logdir = os.path.join(tempfile.TemporaryDirectory().name, "c3logs")
    opt = OptimalControl(
        dir_path=logdir,
        fid_func=unitary_infid_set,
        pmap=exp.pmap,
        fid_subspace=["Q1", "Q2"],
        algorithm=algorithms[request.param],
        options={"maxiters": 2} if request.param == "tf_sgd" else {"maxiter": 2},
        run_name=f"better_X90_{request.param}",
    )
    opt.set_exp(exp)
    return opt
