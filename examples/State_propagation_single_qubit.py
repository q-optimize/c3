#%%
import copy
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from c3.c3objs import Quantity as Qty
from c3.parametermap import ParameterMap as PMap
from c3.experiment import Experiment as Exp
from c3.model import Model as Mdl
from c3.generator.generator import Generator as Gnr

# Building blocks
import c3.generator.devices as devices
import c3.signal.gates as gates
import c3.libraries.chip as chip
import c3.signal.pulse as pulse

# Libs and helpers
import c3.libraries.hamiltonians as hamiltonians
import c3.libraries.envelopes as envelopes


qubit_levels = 4
qubit_frequency = 5e9
qubit_anharm = -200e6
qubit_t1 = 20e-9
qubit_t2star = 40e-9
qubit_temp = 10e-6

qubit = chip.Qubit(
    name="Q",
    desc="Qubit",
    freq=Qty(value=qubit_frequency, min_val=1e9, max_val=8e9, unit="Hz 2pi"),
    anhar=Qty(value=qubit_anharm, min_val=-380e6, max_val=-120e6, unit="Hz 2pi"),
    hilbert_dim=qubit_levels,
    t1=Qty(value=qubit_t1, min_val=1e-9, max_val=90e-3, unit="s"),
    t2star=Qty(value=qubit_t2star, min_val=10e-9, max_val=90e-3, unit="s"),
    temp=Qty(value=qubit_temp, min_val=0.0, max_val=0.12, unit="K"),
)

drive_qubit = chip.Drive(
    name="dQ",
    desc="Qubit Drive 1",
    comment="Drive line on qubit",
    connected=["Q"],
    hamiltonian_func=hamiltonians.x_drive,
)

model = Mdl(
    [qubit],  # Individual, self-contained components
    [drive_qubit],  # Interactions between components
)
model.set_lindbladian(True)
model.set_dressed(False)


sim_res = 100e9
awg_res = 2e9
v2hz = 1e9

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
        "dQ": {
            "LO": [],
            "AWG": [],
            "DigitalToAnalog": ["AWG"],
            "Mixer": ["LO", "DigitalToAnalog"],
            "VoltsToHertz": ["Mixer"],
        }
    },
)


# ### MAKE GATESET
t_final = 7e-9
sideband = 50e6
lo_freq = 5e9 + sideband


gauss_params_single = {
    "amp": Qty(value=0.45, min_val=0.35, max_val=0.6, unit="V"),
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
        value=-sideband - 0.5e6,
        min_val=-60 * 1e6,
        max_val=-40 * 1e6,
        unit="Hz 2pi",
    ),
    "delta": Qty(value=-1, min_val=-5, max_val=3, unit=""),
}

gauss_env_single = pulse.EnvelopeDrag(
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
carrier_parameters = {
    "freq": Qty(
        value=lo_freq,
        min_val=4.5e9,
        max_val=6e9,
        unit="Hz 2pi",
    ),
    "framechange": Qty(value=0.0, min_val=-np.pi, max_val=3 * np.pi, unit="rad"),
}
carr = pulse.Carrier(
    name="carrier",
    desc="Frequency of the local oscillator",
    params=carrier_parameters,
)

rx90p = gates.Instruction(
    name="rx90p", t_start=0.0, t_end=t_final, channels=["dQ"], targets=[0]
)
QId = gates.Instruction(
    name="id", t_start=0.0, t_end=100e-9, channels=["dQ"], targets=[0]
)

rx90p.add_component(gauss_env_single, "dQ")
rx90p.add_component(carr, "dQ")
QId.add_component(nodrive_env, "dQ")
QId.add_component(copy.deepcopy(carr), "dQ")
QId.comps["dQ"]["carrier"].params["framechange"].set_value(
    (-sideband * t_final) % (2 * np.pi)
)
ry90p = copy.deepcopy(rx90p)
ry90p.name = "ry90p"
rx90m = copy.deepcopy(rx90p)
rx90m.name = "rx90m"
ry90m = copy.deepcopy(rx90p)
ry90m.name = "ry90m"
ry90p.comps["dQ"]["gauss"].params["xy_angle"].set_value(0.5 * np.pi)
rx90m.comps["dQ"]["gauss"].params["xy_angle"].set_value(np.pi)
ry90m.comps["dQ"]["gauss"].params["xy_angle"].set_value(1.5 * np.pi)

parameter_map = PMap(
    instructions=[QId, rx90p, ry90p, rx90m, ry90m], model=model, generator=generator
)

# ### MAKE EXPERIMENT
exp = Exp(pmap=parameter_map)

model.set_FR(False)

model.set_lindbladian(True)
psi_init = [[0] * model.tot_dim]
init_state_index = model.get_state_indeces([(1,)])[0]
psi_init[0][init_state_index] = 1
init_state = tf.transpose(tf.constant(psi_init, tf.complex128))
sequence = ["rx90p[0]"]

exp.set_opt_gates(sequence)
model.set_init_state(init_state)

compute_states_tf = tf.function(exp.compute_states)
result = compute_states_tf(solver="rk4")
psis = result["states"]
ts = result["ts"]


pops = []
for rho in psis:
    pops.append(tf.math.real(tf.linalg.diag_part(rho)))

fig, axs = plt.subplots(1, 1)
fig.set_dpi(100)
axs.plot(ts / 1e-9, pops)
axs.grid(linestyle="--")
axs.tick_params(
    direction="in", left=True, right=True, top=True, bottom=True
)
axs.set_xlabel('Time [ns]')
axs.set_ylabel('Population')
plt.legend(model.state_labels)
plt.show()


model.set_lindbladian(True)
psi_init = [[0] * model.tot_dim]
init_state_index = model.get_state_indeces([(1,)])[0]
psi_init[0][init_state_index] = 1
init_state = tf.transpose(tf.constant(psi_init, tf.complex128))
sequence = ["id[0]"]

exp.set_opt_gates(sequence)
model.set_init_state(init_state)

compute_states_tf = tf.function(exp.compute_states)
result = compute_states_tf(solver="rk4")
psis = result["states"]
ts = result["ts"]


pops = []
for rho in psis:
    pops.append(tf.math.real(tf.linalg.diag_part(rho)))

fig, axs = plt.subplots(1, 1)
fig.set_dpi(100)
axs.plot(ts / 1e-9, pops)
axs.grid(linestyle="--")
axs.tick_params(
    direction="in", left=True, right=True, top=True, bottom=True
)
axs.set_xlabel('Time [ns]')
axs.set_ylabel('Population')
plt.legend(model.state_labels)
plt.show()


# %%
