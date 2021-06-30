"""
integration testing module for C1 optimization through two-qubits example
"""

import os
import tempfile
import copy
import pickle
import numpy as np
import pytest
from numpy.testing import assert_array_almost_equal as almost_equal

# Main C3 objects
from c3.c3objs import Quantity as Qty
from c3.parametermap import ParameterMap as Pmap
from c3.experiment import Experiment as Exp
from c3.model import Model as Mdl
from c3.generator.generator import Generator as Gnr

# Building blocks
import c3.generator.devices as devices
import c3.signal.pulse as pulse
import c3.signal.gates as gates

# Libs and helpers
import c3.libraries.algorithms as algorithms
import c3.libraries.chip as chip
import c3.libraries.envelopes as envelopes
import c3.libraries.fidelities as fidelities
import c3.libraries.hamiltonians as hamiltonians
import c3.libraries.tasks as tasks

from c3.optimizers.c1 import C1

logdir = os.path.join(tempfile.TemporaryDirectory().name, "c3logs")

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

init_temp = 50e-3
init_ground = tasks.InitialiseGround(
    init_temp=Qty(value=init_temp, min_val=-0.001, max_val=0.22, unit="K")
)

model = Mdl(
    [q1, q2],  # Individual, self-contained components
    [drive, drive2, q1q2],  # Interactions between components
    [conf_matrix, init_ground],  # SPAM processing
)

model.set_lindbladian(False)
model.set_dressed(True)

sim_res = 100e9  # Resolution for numerical simulation
awg_res = 2e9  # Realistic, limited resolution of an AWG

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
    chains={
        "d1": ["LO", "AWG", "DigitalToAnalog", "Response", "Mixer", "VoltsToHertz"],
        "d2": ["LO", "AWG", "DigitalToAnalog", "Response", "Mixer", "VoltsToHertz"],
    },
)

t_final = 7e-9  # Time for single qubit gates
sideband = 50e6
gauss_params_single = {
    "amp": Qty(value=0.5, min_val=0.4, max_val=0.6, unit="V"),
    "t_final": Qty(
        value=t_final, min_val=0.5 * t_final, max_val=1.5 * t_final, unit="s"
    ),
    "sigma": Qty(value=t_final / 4, min_val=t_final / 8, max_val=t_final / 2, unit="s"),
    "xy_angle": Qty(value=0.0, min_val=-0.5 * np.pi, max_val=2.5 * np.pi, unit="rad"),
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
    name="carrier", desc="Frequency of the local oscillator", params=carrier_parameters
)

lo_freq_q2 = 5.6e9 + sideband
carr_2 = copy.deepcopy(carr)
carr_2.params["freq"].set_value(lo_freq_q2)

rx90p_q1 = gates.Instruction(
    name="rx90p",
    targets=[0],
    t_start=0.0,
    t_end=t_final,
    channels=["d1", "d2"],
    params={"use_t_before": True},
)
rx90p_q2 = gates.Instruction(
    name="rx90p",
    targets=[1],
    t_start=0.0,
    t_end=t_final,
    channels=["d1", "d2"],
    params={"use_t_before": True},
)
QId_q1 = gates.Instruction(
    name="id",
    targets=[0],
    t_start=0.0,
    t_end=t_final,
    channels=["d1", "d2"],
    params={"use_t_before": True},
)
QId_q2 = gates.Instruction(
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

pmap = Pmap(single_q_gates, generator, model)

exp = Exp(pmap)

generator.devices["AWG"].enable_drag_2()

exp.set_opt_gates(["rx90p[0]"])

gateset_opt_map = [
    [
        ("rx90p[0]", "d1", "gauss", "amp"),
    ],
    [
        ("rx90p[0]", "d1", "gauss", "freq_offset"),
    ],
    [
        ("rx90p[0]", "d1", "gauss", "xy_angle"),
    ],
    [
        ("rx90p[0]", "d1", "gauss", "delta"),
    ],
]

pmap.set_opt_map(gateset_opt_map)

opt = C1(
    dir_path=logdir,
    fid_func=fidelities.average_infid_set,
    fid_subspace=["Q1", "Q2"],
    pmap=pmap,
    algorithm=algorithms.tf_sgd,
    options={"maxiters": 5},
    run_name="better_X90_tf_sgd",
)

opt.set_exp(exp)

with open("test/two_qubit_data.pickle", "rb") as filename:
    test_data = pickle.load(filename)

gen_signal = generator.generate_signals(pmap.instructions["rx90p[0]"])
ts = gen_signal["d1"]["ts"]
hdrift, hks = model.get_Hamiltonians()
propagator = exp.propagation(gen_signal, "rx90p[0]")


def test_signals() -> None:
    np.testing.assert_allclose(ts, test_data["ts"])
    np.testing.assert_allclose(
        actual=gen_signal["d1"]["values"].numpy(),
        desired=test_data["signal"]["d1"]["values"].numpy(),
    )


def test_hamiltonians() -> None:
    assert (hdrift.numpy() - test_data["hdrift"].numpy() < 1).any()
    for key in hks:
        almost_equal(hks[key], test_data["hks"][key])


def test_propagation() -> None:
    almost_equal(propagator, test_data["propagator"])


@pytest.mark.slow
@pytest.mark.tensorflow
@pytest.mark.optimizers
@pytest.mark.integration
def test_optim_tf_sgd() -> None:
    """
    check if optimization result is below 1e-2
    """
    opt.store_unitaries = True
    opt.optimize_controls()
    assert opt.current_best_goal < 0.01


@pytest.mark.optimizers
@pytest.mark.slow
@pytest.mark.integration
@pytest.mark.tensorflow
def test_bad_tf_sgd() -> None:
    bad_tf_opt = C1(
        dir_path=logdir,
        fid_func=fidelities.average_infid_set,
        fid_subspace=["Q1", "Q2"],
        pmap=pmap,
        algorithm=algorithms.tf_sgd,
        options={"maxfun": 2},
        run_name="better_X90_bad_tf",
    )
    bad_tf_opt.set_exp(exp)

    with pytest.raises(KeyError):
        bad_tf_opt.optimize_controls()


@pytest.mark.optimizers
@pytest.mark.slow
@pytest.mark.integration
def test_optim_lbfgs() -> None:
    lbfgs_opt = C1(
        dir_path=logdir,
        fid_func=fidelities.average_infid_set,
        fid_subspace=["Q1", "Q2"],
        pmap=pmap,
        algorithm=algorithms.lbfgs,
        options={"maxfun": 2},
        run_name="better_X90_lbfgs",
    )
    lbfgs_opt.set_exp(exp)

    lbfgs_opt.optimize_controls()
    assert lbfgs_opt.current_best_goal < 0.01


@pytest.mark.optimizers
@pytest.mark.slow
@pytest.mark.integration
def test_optim_lbfgs_grad_free() -> None:
    lbfgs_grad_free_opt = C1(
        dir_path=logdir,
        fid_func=fidelities.average_infid_set,
        fid_subspace=["Q1", "Q2"],
        pmap=pmap,
        algorithm=algorithms.lbfgs_grad_free,
        options={"maxfun": 5},
        run_name="grad_free_lbfgs",
    )
    lbfgs_grad_free_opt.set_exp(exp)

    lbfgs_grad_free_opt.optimize_controls()
    assert lbfgs_grad_free_opt.current_best_goal < 0.01
