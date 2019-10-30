"""Simple propagation."""

import c3po.hamiltonians as hamiltonians
import c3po.envelopes as envelopes
import c3po.control as control
from c3po.simulator import Simulator as Sim
import numpy as np
import tensorflow as tf
from IBM_1q_chip import create_chip_model, create_generator

# System
qubit_freq = 5.1173e9 * 2 * np.pi
qubit_anhar = -3155137343 * 2 * np.pi
qubit_lvls = 6
drive_ham = hamiltonians.x_drive
v_hz_conversion = 1e9 * 0.40605
t_final = 10e-9

# Simulation variables
sim_res = 1e11  # 100GHz
awg_res = 1e9  # GHz

# Create system
model = create_chip_model(qubit_freq, qubit_anhar, qubit_lvls, drive_ham)
gen = create_generator(sim_res, awg_res, v_hz_conversion)
gen.devices['awg'].options = 'pwc'

values = np.array(
            [0.24638675-0.23929157j, 0.21741223-0.21632876j,
             0.37322429-0.01629624j, 0.35744771-0.14401254j,
             0.72712549-0.31812883j, 0.51919636+0.20880559j,
             0.68819932+0.03360306j, 0.15161798+0.01600276j,
             0.11390550+0.01708678j, 0.52968099+0.12146439j]
)

# Gates
pwc_params = {
    'inphase': np.real(values),
    'quadrature': np.imag(values)
}

pwc_bounds = {
    'inphase': [0, 0]*len(values),
    'quadrature': [0, 0]*len(values)
    }

carrier_parameters = {
    'freq': qubit_freq
}
carrier_bounds = {
    'freq': [5e9 * 2 * np.pi, 7e9 * 2 * np.pi]
}
pwc_env = control.Envelope(
    name="gauss",
    desc="Gaussian comp 1 of signal 1",
    params=pwc_params,
    bounds=pwc_bounds,
    shape=envelopes.pwc
)
carr = control.Carrier(
    name="carrier",
    desc="Frequency of the local oscillator",
    params=carrier_parameters,
    bounds=carrier_bounds
)
X90p = control.Instruction(
    name="X90p",
    t_start=0.0,
    t_end=t_final,
    channels=["d1"]
)
X90p.add_component(pwc_env, "d1")
X90p.add_component(carr, "d1")
gates = control.GateSet()
gates.add_instruction(X90p)

# Simulation class and fidelity function
sim = Sim(model, gen, gates)
opt_map = gates.list_parameters()

pulse_values, _ = gates.get_parameters(opt_map)
model_params, _ = model.get_values_bounds()
signal = gen.generate_signals(gates.instructions["X90p"])
U = sim.propagation(signal, model_params)

# plot dynamics
qubit_g = np.zeros([qubit_lvls, 1])
qubit_g[0] = 1
ket_0 = tf.constant(qubit_g, tf.complex128)
sim.plot_dynamics(ket_0)
