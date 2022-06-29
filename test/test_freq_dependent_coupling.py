import tensorflow as tf
import numpy as np
import pytest


from c3.c3objs import Quantity as Qty
import c3.libraries.hamiltonians as hamiltonians
import c3.utils.qt_utils as qt_utils
from c3.libraries.chip import Coupling_Drive
from c3.generator.devices import CouplingTuning


@pytest.mark.unit
def test_driven_coupling() -> None:
    Coup_drive = Coupling_Drive(
        name="Cd", connected=["Q1", "Q2"], hamiltonian_func=hamiltonians.int_YY
    )

    a = np.array([[0, 1], [0, 0]])
    a0 = qt_utils.hilbert_space_kron(a, 0, [2, 2])
    a1 = qt_utils.hilbert_space_kron(a, 1, [2, 2])

    Coup_drive.init_Hs([a0, a1])

    drive_ham = Coup_drive.h
    expected_ham = tf.constant(
        [
            [-0.0 + 0.0j, -0.0 + 0.0j, -0.0 + 0.0j, -1.0 + 0.0j],
            [-0.0 + 0.0j, -0.0 + 0.0j, 1.0 + 0.0j, -0.0 + 0.0j],
            [-0.0 + 0.0j, 1.0 + 0.0j, -0.0 + 0.0j, -0.0 + 0.0j],
            [-1.0 + 0.0j, -0.0 + 0.0j, -0.0 + 0.0j, -0.0 + 0.0j],
        ],
        dtype=tf.complex128,
    )

    assert np.all(drive_ham == expected_ham)


@pytest.mark.unit
def test_frequency_dependent_coupling() -> None:
    # Check with one frequency modulation
    freq_q1 = 5e9
    freq_q2 = 6e9
    coupling_strength = 100e6

    wd = 500e6
    t_final = 100e-9
    ts = np.linspace(0, t_final, 10)
    freq_mod_q1 = np.sin(wd * ts)

    signal_in = [{"ts": ts, "values": freq_mod_q1 * 2 * np.pi}]

    coup_tuning_1 = CouplingTuning(
        name="coup_tuning_one",
        two_inputs=False,
        w_1=Qty(freq_q1, unit="Hz 2pi"),
        w_2=Qty(freq_q2, unit="Hz 2pi"),
        g0=Qty(coupling_strength, unit="Hz 2pi"),
    )

    signal_out = coup_tuning_1.process(None, "", signal_in)
    expected_values = (
        coupling_strength
        / np.sqrt(freq_q1 * freq_q2)
        * np.sqrt((freq_q1 - freq_mod_q1) * freq_q2)
        - coupling_strength
    )

    assert np.all(signal_out["ts"] == ts)
    assert np.all(np.isclose(signal_out["values"] / (2 * np.pi), expected_values))

    # Check with two frequency modulations
    wd2 = 200e6
    freq_mod_q2 = np.sin(wd2 * ts)

    signal_in2 = [
        {"ts": ts, "values": freq_mod_q1 * 2 * np.pi},
        {"ts": ts, "values": freq_mod_q2 * 2 * np.pi},
    ]

    coup_tuning_2 = CouplingTuning(
        name="coup_tuning_one",
        two_inputs=True,
        w_1=Qty(freq_q1, unit="Hz 2pi"),
        w_2=Qty(freq_q2, unit="Hz 2pi"),
        g0=Qty(coupling_strength, unit="Hz 2pi"),
    )
    signal_out2 = coup_tuning_2.process(None, "", signal_in2)
    expected_values2 = (
        coupling_strength
        / np.sqrt(freq_q1 * freq_q2)
        * np.sqrt((freq_q1 - freq_mod_q1) * (freq_q2 - freq_mod_q2))
        - coupling_strength
    )

    assert np.all(signal_out2["ts"] == ts)
    assert np.all(np.isclose(signal_out2["values"] / (2 * np.pi), expected_values2))
