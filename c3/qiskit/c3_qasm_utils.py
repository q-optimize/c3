"""Convenience Module for creating different c3 components
c3_qasm_simulator
"""

from typing import List, Tuple
from qiskit import qobj
import tensorflow as tf
import copy
import numpy as np

# Main C3 objects
from c3.c3objs import Quantity as Qty
from c3.parametermap import ParameterMap as Pmap
from c3.experiment import Experiment as Exp
from c3.system.model import Model as Mdl
from c3.generator.generator import Generator as Gnr

# Building blocks
import c3.generator.devices as devices
import c3.system.chip as chip
import c3.signal.pulse as pulse
import c3.signal.gates as gates
import c3.system.tasks as tasks

# Libs and helpers
import c3.libraries.algorithms as algorithms
import c3.libraries.hamiltonians as hamiltonians
import c3.libraries.fidelities as fidelities
import c3.libraries.envelopes as envelopes


def get_perfect_qubits(n_qubits: int) -> List[chip.Qubit]:
    """Instantiate and return a list of perfect C3 qubits

    Parameters
    ----------
    n_qubits : int
        Number of qubits to be returned

    Returns
    -------
    List[chip.Qubit]
        A list of perfect qubits
    """
    qubit_lvls = 3
    freq_q1 = 5e9
    anhar_q1 = -210e6
    t1_q1 = 27e-6
    t2star_q1 = 39e-6
    qubit_temp = 50e-3

    q1 = chip.Qubit(
        name="Q1",
        desc="Qubit 1",
        freq=Qty(value=freq_q1, min_val=4.995e9, max_val=5.005e9, unit="Hz 2pi"),
        anhar=Qty(value=anhar_q1, min_val=-380e6, max_val=-120e6, unit="Hz 2pi"),
        hilbert_dim=qubit_lvls,
        t1=Qty(value=t1_q1, min_val=1e-6, max_val=90e-6, unit="s"),
        t2star=Qty(value=t2star_q1, min_val=10e-6, max_val=90e-3, unit="s"),
        temp=Qty(value=qubit_temp, min_val=0.0, max_val=0.12, unit="K"),
    )

    freq_q2 = 5.6e9
    anhar_q2 = -240e6
    t1_q2 = 23e-6
    t2star_q2 = 31e-6
    q2 = chip.Qubit(
        name="Q2",
        desc="Qubit 2",
        freq=Qty(value=freq_q2, min_val=5.595e9, max_val=5.605e9, unit="Hz 2pi"),
        anhar=Qty(value=anhar_q2, min_val=-380e6, max_val=-120e6, unit="Hz 2pi"),
        hilbert_dim=qubit_lvls,
        t1=Qty(value=t1_q2, min_val=1e-6, max_val=90e-6, unit="s"),
        t2star=Qty(value=t2star_q2, min_val=10e-6, max_val=90e-6, unit="s"),
        temp=Qty(value=qubit_temp, min_val=0.0, max_val=0.12, unit="K"),
    )

    return [q1, q2]


def get_coupling_fc(n_qubits: int) -> List[chip.Coupling]:
    """Instantiate and return a fully connected coupling

    Parameters
    ----------
    n_qubits : int
        Number of qubits in the system

    Returns
    -------
    List[chip.Coupling]
        A fully connected coupling map
    """
    coupling_strength = 20e6
    q1q2 = chip.Coupling(
        name="Q1-Q2",
        desc="coupling",
        comment="Coupling qubit 1 to qubit 2",
        connected=["Q1", "Q2"],
        strength=Qty(
            value=coupling_strength, min_val=-1 * 1e3, max_val=200e6, unit="Hz 2pi"
        ),
        hamiltonian_func=hamiltonians.int_XX,
    )
    return [q1q2]


def get_drives(n_qubits: int) -> List[chip.Drive]:
    """Instantiate and return drives for all qubits

    Parameters
    ----------
    n_qubits : int
        Number of qubits in the system

    Returns
    -------
    List[chip.Drive]
        List of drives for all the qubits
    """
    drive = chip.Drive(
        name="d1",
        desc="Drive 1",
        comment="Drive line 1 on qubit 1",
        connected=["Q1"],
        hamiltonian_func=hamiltonians.x_drive,
    )
    drive2 = chip.Drive(
        name="d2",
        desc="Drive 2",
        comment="Drive line 2 on qubit 2",
        connected=["Q2"],
        hamiltonian_func=hamiltonians.x_drive,
    )
    return [drive, drive2]


def get_confusion_no_spam(n_qubits: int) -> tasks.ConfusionMatrix:
    """Return a dummy confusion matrix with no spam errors

    Parameters
    ----------
    n_qubits : int
        Number of qubits in the system

    Returns
    -------
    tasks.ConfusionMatrix
        No Spam errors Confusion Matrix
    """
    qubit_lvls = 3
    m00_q1 = 0.97  # Prop to read qubit 1 state 0 as 0
    m01_q1 = 0.04  # Prop to read qubit 1 state 0 as 1
    m00_q2 = 0.96  # Prop to read qubit 2 state 0 as 0
    m01_q2 = 0.05  # Prop to read qubit 2 state 0 as 1
    one_zeros = np.array([0] * qubit_lvls)
    zero_ones = np.array([1] * qubit_lvls)
    one_zeros[0] = 1
    zero_ones[0] = 0
    val1 = one_zeros * m00_q1 + zero_ones * m01_q1
    val2 = one_zeros * m00_q2 + zero_ones * m01_q2
    min_val = one_zeros * 0.8 + zero_ones * 0.0
    max_val = one_zeros * 1.0 + zero_ones * 0.2
    confusion_row1 = Qty(value=val1, min_val=min_val, max_val=max_val, unit="")
    confusion_row2 = Qty(value=val2, min_val=min_val, max_val=max_val, unit="")
    conf_matrix = tasks.ConfusionMatrix(Q1=confusion_row1, Q2=confusion_row2)

    return [conf_matrix]


def get_init_thermal_state() -> tasks.InitialiseGround:
    """Return an Initial Thermal State at 50 nK

    Returns
    -------
    tasks.InitialiseGround
        Thermal State at 50 nK
    """
    init_temp = 50e-3
    init_ground = tasks.InitialiseGround(
        init_temp=Qty(value=init_temp, min_val=-0.001, max_val=0.22, unit="K")
    )
    return [init_ground]


def get_generator() -> Gnr:
    """C3 Model for an Generator

    Returns
    -------
    Gnr
        C3 Generator object
    """
    sim_res = 100e9  # Resolution for numerical simulation
    awg_res = 2e9  # Realistic, limited resolution of an AWG
    lo = devices.LO(name="lo", resolution=sim_res)
    awg = devices.AWG(name="awg", resolution=awg_res)
    mixer = devices.Mixer(name="mixer")

    resp = devices.Response(
        name="resp",
        rise_time=Qty(value=0.3e-9, min_val=0.05e-9, max_val=0.6e-9, unit="s"),
        resolution=sim_res,
    )

    dig_to_an = devices.DigitalToAnalog(name="dac", resolution=sim_res)

    v2hz = 1e9
    v_to_hz = devices.VoltsToHertz(
        name="v_to_hz",
        V_to_Hz=Qty(value=v2hz, min_val=0.9e9, max_val=1.1e9, unit="Hz/V"),
    )

    generator = Gnr(
        devices={
            "LO": devices.LO(name="lo", resolution=sim_res, outputs=1),
            "AWG": devices.AWG(name="awg", resolution=awg_res, outputs=1),
            "DigitalToAnalog": devices.DigitalToAnalog(
                name="dac", resolution=sim_res, inputs=1, outputs=1
            ),
            "Response": devices.Response(
                name="resp",
                rise_time=Qty(value=0.3e-9, min_val=0.05e-9, max_val=0.6e-9, unit="s"),
                resolution=sim_res,
                inputs=1,
                outputs=1,
            ),
            "Mixer": devices.Mixer(name="mixer", inputs=2, outputs=1),
            "VoltsToHertz": devices.VoltsToHertz(
                name="v_to_hz",
                V_to_Hz=Qty(value=1e9, min_val=0.9e9, max_val=1.1e9, unit="Hz/V"),
                inputs=1,
                outputs=1,
            ),
        },
        chain=["LO", "AWG", "DigitalToAnalog", "Response", "Mixer", "VoltsToHertz"],
    )

    return generator


def get_gate_set(n_qubits: int) -> List[gates.Instruction]:
    """List of single and 2-qubit gates for all qubits

    Parameters
    ----------
    n_qubits : int
        Number of qubits in the system

    Returns
    -------
    List[gates.Instruction]
        List of u3, cx, id and unitary gates for all qubits
    """
    t_final = 7e-9  # Time for single qubit gates
    sideband = 50e6
    gauss_params_single = {
        "amp": Qty(value=0.5, min_val=0.4, max_val=0.6, unit="V"),
        "t_final": Qty(
            value=t_final, min_val=0.5 * t_final, max_val=1.5 * t_final, unit="s"
        ),
        "sigma": Qty(
            value=t_final / 4, min_val=t_final / 8, max_val=t_final / 2, unit="s"
        ),
        "xy_angle": Qty(
            value=0.0, min_val=-0.5 * np.pi, max_val=2.5 * np.pi, unit="rad"
        ),
        "freq_offset": Qty(
            value=-sideband - 3e6, min_val=-56 * 1e6, max_val=-52 * 1e6, unit="Hz 2pi"
        ),
        "delta": Qty(value=-1, min_val=-5, max_val=3, unit=""),
    }

    gauss_env_single = pulse.Envelope(
        name="gauss",
        desc="Gaussian comp for single-qubit gates",
        params=gauss_params_single,
        shape=envelopes.gaussian_nonorm,
    )

    nodrive_env = pulse.Envelope(
        name="no_drive",
        params={
            "t_final": Qty(
                value=t_final, min_val=0.5 * t_final, max_val=1.5 * t_final, unit="s"
            )
        },
        shape=envelopes.no_drive,
    )

    lo_freq_q1 = 5e9 + sideband
    carrier_parameters = {
        "freq": Qty(value=lo_freq_q1, min_val=4.5e9, max_val=6e9, unit="Hz 2pi"),
        "framechange": Qty(value=0.0, min_val=-np.pi, max_val=3 * np.pi, unit="rad"),
    }
    carr = pulse.Carrier(
        name="carrier",
        desc="Frequency of the local oscillator",
        params=carrier_parameters,
    )

    lo_freq_q2 = 5.6e9 + sideband
    carr_2 = copy.deepcopy(carr)
    carr_2.params["freq"].set_value(lo_freq_q2)

    X90p_q1 = gates.Instruction(
        name="X90p", t_start=0.0, t_end=t_final, channels=["d1"]
    )
    X90p_q2 = gates.Instruction(
        name="X90p", t_start=0.0, t_end=t_final, channels=["d2"]
    )
    QId_q1 = gates.Instruction(name="Id", t_start=0.0, t_end=t_final, channels=["d1"])
    QId_q2 = gates.Instruction(name="Id", t_start=0.0, t_end=t_final, channels=["d2"])

    X90p_q1.add_component(gauss_env_single, "d1")
    X90p_q1.add_component(carr, "d1")
    QId_q1.add_component(nodrive_env, "d1")
    QId_q1.add_component(copy.deepcopy(carr), "d1")

    X90p_q2.add_component(copy.deepcopy(gauss_env_single), "d2")
    X90p_q2.add_component(carr_2, "d2")
    QId_q2.add_component(copy.deepcopy(nodrive_env), "d2")
    QId_q2.add_component(copy.deepcopy(carr_2), "d2")

    Y90p_q1 = copy.deepcopy(X90p_q1)
    Y90p_q1.name = "Y90p"
    X90m_q1 = copy.deepcopy(X90p_q1)
    X90m_q1.name = "X90m"
    Y90m_q1 = copy.deepcopy(X90p_q1)
    Y90m_q1.name = "Y90m"
    Y90p_q1.comps["d1"]["gauss"].params["xy_angle"].set_value(0.5 * np.pi)
    X90m_q1.comps["d1"]["gauss"].params["xy_angle"].set_value(np.pi)
    Y90m_q1.comps["d1"]["gauss"].params["xy_angle"].set_value(1.5 * np.pi)
    Q1_gates = [QId_q1, X90p_q1, Y90p_q1, X90m_q1, Y90m_q1]

    Y90p_q2 = copy.deepcopy(X90p_q2)
    Y90p_q2.name = "Y90p"
    X90m_q2 = copy.deepcopy(X90p_q2)
    X90m_q2.name = "X90m"
    Y90m_q2 = copy.deepcopy(X90p_q2)
    Y90m_q2.name = "Y90m"
    Y90p_q2.comps["d2"]["gauss"].params["xy_angle"].set_value(0.5 * np.pi)
    X90m_q2.comps["d2"]["gauss"].params["xy_angle"].set_value(np.pi)
    Y90m_q2.comps["d2"]["gauss"].params["xy_angle"].set_value(1.5 * np.pi)
    Q2_gates = [QId_q2, X90p_q2, Y90p_q2, X90m_q2, Y90m_q2]

    all_1q_gates_comb = []
    for g1 in Q1_gates:
        for g2 in Q2_gates:
            g = gates.Instruction(name="NONE", t_start=0.0, t_end=t_final, channels=[])
            g.name = g1.name + ":" + g2.name
            channels = []
            channels.extend(g1.comps.keys())
            channels.extend(g2.comps.keys())
            for chan in channels:
                g.comps[chan] = {}
                if chan in g1.comps:
                    g.comps[chan].update(g1.comps[chan])
                if chan in g2.comps:
                    g.comps[chan].update(g2.comps[chan])
            all_1q_gates_comb.append(g)

    return all_1q_gates_comb


def get_opt_gates(n_qubits: int) -> List[str]:
    """Return the list of gates to optimize

    Parameters
    ----------
    n_qubits : int
        Number of qubits in system

    Returns
    -------
    List[str]
        Dummy list containing all gates
    """
    return ["X90p:Id", "Id:Id"]


def get_gateset_opt_map(n_qubits: int) -> List[List[Tuple]]:
    """Return parameter map to optimize

    Parameters
    ----------
    n_qubits : int
        Number of qubits in the system

    Returns
    -------
    List[List[Tuple]]
        Dummy list of gate parameters to optimize
    """
    pass


def get_sequence(instructions: dict) -> List[str]:
    """Return a sequence of gates from instructions

    Parameters
    ----------
    instructions : dict
        Instructions from the qasm experiment

    Returns
    -------
    List[str]
        List of gates
    """
    # TODO conditional
    # conditional = getattr(instructions, "conditional", None)  # noqa

    # TODO unitary

    # TODO U, u3

    # TODO CX, cx

    # TODO id, u0

    # TODO reset

    # TODO barrier

    # TODO measure

    # TODO binary function

    # TODO raise C3QiskitError if unknown instruction
    pass


def get_init_ground_state(n_qubits: int) -> tf.Tensor:
    """Return a perfect ground state

    Parameters
    ----------
    n_qubits : int
        Number of qubits in the system

    Returns
    -------
    tf.Tensor
        Tensor array of ground state
        shape(2^n, 1), dtype=complex128
    """
    psi_init = [[0] * 9]
    psi_init[0][0] = 1
    init_state = tf.transpose(tf.constant(psi_init, tf.complex128))

    return init_state
