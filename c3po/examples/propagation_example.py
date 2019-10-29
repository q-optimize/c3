"""Simple propagation."""

import c3po.hamiltonians as hamiltonians
from c3po.simulator import Simulator as Sim
import numpy as np
from single_qubit import create_chip_model, create_generator, create_gates

# System
qubit_freq = 6e9 * 2 * np.pi
qubit_anhar = -100e6 * 2 * np.pi
qubit_lvls = 6
drive_ham = hamiltonians.x_drive
v_hz_conversion = 2e9 * np.pi
t_final = 10e-9

# Simulation variables
sim_res = 1e11  # 100GHz
awg_res = 1e9  # GHz

# Create system
model = create_chip_model(qubit_freq, qubit_anhar, qubit_lvls, drive_ham)
gen = create_generator(sim_res, awg_res, v_hz_conversion)
gates = create_gates(t_final, v_hz_conversion, qubit_freq)

# Simulation class and fidelity function
sim = Sim(model, gen, gates)
opt_map = gates.list_parameters()

pulse_values, _ = gates.get_parameters(opt_map)
model_params, _ = model.get_parameters()
signal = gen.generate_signals(gates.instructions["X90p"])
U = sim.propagation(signal, model_params)
