"""
testing module for lindblad propagation
"""

import pytest
import copy
import numpy as np
from c3.model import Model as Mdl
from c3.c3objs import Quantity as Qty
from c3.parametermap import ParameterMap as PMap
from c3.experiment import Experiment as Exp
from c3.generator.generator import Generator as Gnr
import c3.signal.gates as gates
import c3.libraries.chip as chip
import c3.generator.devices as devices
import c3.libraries.hamiltonians as hamiltonians
import c3.signal.pulse as pulse
import c3.libraries.envelopes as envelopes
import c3.libraries.tasks as tasks
import c3.utils.tf_utils as tf_utils

lindblad = True
dressed = True
qubit_lvls = 2
freq = 5e9
anhar = -210e6
init_temp = 0
qubit_temp = 1e-10
t_final = 12e-9
sim_res = 100e9
awg_res = 2e9
lo_freq = freq
qubit_t1 = 0.20e-6
qubit_t2star = 0.39e-6

# ### MAKE MODEL
q1 = chip.Qubit(
    name="Q1",
    desc="Qubit 1",
    freq=Qty(
        value=freq,
        min_val=4.995e9,
        max_val=5.005e9,
        unit="Hz 2pi",
    ),
    anhar=Qty(
        value=anhar,
        min_val=-380e6,
        max_val=-120e6,
        unit="Hz 2pi",
    ),
    hilbert_dim=qubit_lvls,
    temp=Qty(value=qubit_temp, min_val=0.0, max_val=0.12, unit="K"),
    t1=Qty(value=qubit_t1, min_val=0.001e-6, max_val=90e-6, unit="s"),
    t2star=Qty(value=qubit_t2star, min_val=0.001e-6, max_val=90e-6, unit="s"),
)

drive = chip.Drive(
    name="d1",
    desc="Drive 1",
    comment="Drive line 1 on qubit 1",
    connected=["Q1"],
    hamiltonian_func=hamiltonians.x_drive,
)
phys_components = [q1]
line_components = [drive]

init_ground = tasks.InitialiseGround(
    init_temp=Qty(value=init_temp, min_val=-0.001, max_val=0.22, unit="K")
)
task_list = [init_ground]
model = Mdl(phys_components, line_components, task_list)
model.set_lindbladian(lindblad)
model.set_dressed(dressed)

# ### MAKE GENERATOR

generator = Gnr(
    devices={
        "LO": devices.LO(name="lo", resolution=sim_res, outputs=1),
        "AWG": devices.AWG(name="awg", resolution=awg_res, outputs=1),
        "DigitalToAnalog": devices.DigitalToAnalog(
            name="dac", resolution=sim_res, inputs=1, outputs=1
        ),
        "Mixer": devices.Mixer(name="mixer", inputs=2, outputs=1),
        "VoltsToHertz": devices.VoltsToHertz(
            name="v_to_hz",
            V_to_Hz=Qty(value=1e9, min_val=0.9e9, max_val=1.1e9, unit="Hz/V"),
            inputs=1,
            outputs=1,
        ),
    },
)

generator.set_chains(
    {
        "d1": {
            "LO": [],
            "AWG": [],
            "DigitalToAnalog": ["AWG"],
            "Mixer": ["LO", "DigitalToAnalog"],
            "VoltsToHertz": ["Mixer"],
        }
    }
)


gauss_params_single = {
    "amp": Qty(value=0.440936, min_val=0.01, max_val=0.99, unit="V"),
    "t_final": Qty(
        value=t_final, min_val=0.5 * t_final, max_val=1.5 * t_final, unit="s"
    ),
    "sigma": Qty(value=t_final / 4, min_val=t_final / 8, max_val=t_final / 2, unit="s"),
    "xy_angle": Qty(
        value=747.256e-6, min_val=-0.5 * np.pi, max_val=2.5 * np.pi, unit="rad"
    ),
    "freq_offset": Qty(
        value=145.022e3,
        min_val=-10 * 1e6,
        max_val=10 * 1e6,
        unit="Hz 2pi",
    ),
}

nodrive_env = pulse.EnvelopeDrag(
    name="no_drive",
    params={
        "t_final": Qty(
            value=t_final, min_val=0.5 * t_final, max_val=1.5 * t_final, unit="s"
        )
    },
    shape=envelopes.no_drive,
)
carrier_parameters = {
    "freq": Qty(
        value=lo_freq,
        min_val=4.5e9,
        max_val=6e9,
        unit="Hz 2pi",
    ),
    "framechange": Qty(value=0.001493, min_val=-np.pi, max_val=3 * np.pi, unit="rad"),
}
carr = pulse.Carrier(
    name="carrier",
    desc="Frequency of the local oscillator",
    params=carrier_parameters,
)


QId = gates.Instruction(
    name="id", t_start=0.0, t_end=t_final, channels=["d1"], targets=[0]
)

QId.add_component(nodrive_env, "d1")
QId.add_component(copy.deepcopy(carr), "d1")
QId.comps["d1"]["carrier"].params["framechange"].set_value((t_final) % (2 * np.pi))


parameter_map = PMap(instructions=[QId], model=model, generator=generator)

exp = Exp(pmap=parameter_map)

init_dm = np.zeros([parameter_map.model.tot_dim, parameter_map.model.tot_dim])
init_dm[1][1] = 1
init_vec = tf_utils.tf_dm_to_vec(init_dm)
seq = ["id[0]"]
exp.set_opt_gates(seq)


@pytest.mark.unit
def test_dissipation() -> None:
    """Test dissipative nature of lindblad evolution"""

    exp.set_prop_method("pwc")
    unitaries = exp.compute_propagators()
    U = np.array(unitaries["id[0]"])
    final_vec = np.dot(U, init_vec)
    final_pops = exp.populations(final_vec, model.lindbladian)
    assert final_pops[1] < 1


@pytest.mark.unit
def test_t1() -> None:
    """Test that T1 decay corresponds to 1/e the initial |1>-population"""
    n = np.int(qubit_t1 / t_final)
    n += 1
    exp.set_prop_method("pwc")
    unitaries = exp.compute_propagators()
    U = np.array(unitaries["id[0]"])
    final_vec = np.dot(U**n, init_vec)
    final_pops = exp.populations(final_vec, model.lindbladian)
    assert final_pops[1] < (1 / np.exp(1))


init_dm = 0.5 * np.ones([parameter_map.model.tot_dim, parameter_map.model.tot_dim])
init_vec = tf_utils.tf_dm_to_vec(init_dm)


@pytest.mark.unit
def test_t2() -> None:
    """Test that T2 decay corresponds to 1/e the initial coherences"""
    t2 = ((2 * qubit_t1) ** (-1) + (qubit_t2star) ** (-1)) ** (-1)
    n = np.int(t2 / t_final)
    n += 1
    exp.set_prop_method("pwc")
    unitaries = exp.compute_propagators()
    U = np.array(unitaries["id[0]"])
    final_vec = np.dot(U**n, init_vec)
    final_dm = tf_utils.tf_vec_to_dm(final_vec)
    assert np.abs(final_dm[0][1]) < (1 / (2 * np.exp(1)))
