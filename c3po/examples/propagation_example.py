"""Simple propagation."""

import tensorflow as tf
from c3po.utils import log_setup
import c3po.hamiltonians as hamiltonians
from c3po.simulator import Simulator as Sim
from c3po.experiment import Experiment as Exp
from c3po.qt_utils import basis
import numpy as np
from single_qubit import create_chip_model, create_generator, create_gates

# System
qubit_freq = 5e9 * 2 * np.pi
qubit_anhar = -300e6 * 2 * np.pi
qubit_lvls = 6
drive_ham = hamiltonians.x_drive
v_hz_conversion = 1  # 2e9 * np.pi
t_final = 3e-9
logdir = log_setup("/tmp/c3logs/")
ket_0 = tf.constant(basis(qubit_lvls, 0), dtype=tf.complex128)

# Simulation variables
sim_res = 1e11  # 100GHz
awg_res = 1e9  # GHz

# Create system
model = create_chip_model(qubit_freq, qubit_anhar, qubit_lvls, drive_ham)
gen = create_generator(sim_res, awg_res, v_hz_conversion, logdir=logdir)
gates = create_gates(t_final, v_hz_conversion, qubit_freq, qubit_anhar)

# Simulation class and fidelity function
exp = Exp(model, gen)
sim = Sim(exp, gates)

signal = gen.generate_signals(gates.instructions["X90p"])
U = sim.propagation(signal)

gateset_opt_map = [[('X90p', 'd1', 'gauss', 'amp')]]
params = gates.get_parameters(gateset_opt_map)
U_dict = sim.get_gates(params, gateset_opt_map)
r, A, B = sim.RB(ket_0, U_dict, num_seqs=20)
