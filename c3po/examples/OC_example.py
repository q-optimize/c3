"""General clean OC script."""

import numpy as np
import tensorflow as tf
import c3po.hamiltonians as hamiltonians
from c3po.utils import log_setup
from c3po.simulator import Simulator as Sim
from c3po.optimizer import Optimizer as Opt
from c3po.experiment import Experiment as Exp
from single_qubit import create_chip_model, create_generator, create_gates

logdir = log_setup("/tmp/c3logs/")

# System
qubit_freq = 6e9 * 2 * np.pi
qubit_anhar = -300e6 * 2 * np.pi
qubit_lvls = 4
drive_ham = hamiltonians.x_drive
v_hz_conversion = 2e9 * np.pi
t_final = 10e-9

# Simulation variables
sim_res = 60e9
awg_res = 1e9  # 1.2GHz

# Create system
model = create_chip_model(qubit_freq, qubit_anhar, qubit_lvls, drive_ham)
gen = create_generator(sim_res, awg_res, v_hz_conversion, logdir=logdir)
gates = create_gates(t_final, v_hz_conversion, qubit_freq, qubit_anhar)

gen.devices['awg'].options = 'drag'

# Simulation class and fidelity function
exp = Exp(model, gen)
sim = Sim(exp, gates)

# Define states
psi_g = np.zeros([qubit_lvls, 1])
psi_g[0] = 1
psi_e = np.zeros([qubit_lvls, 1])
psi_e[1] = 1
psi_ym = (psi_g - 1.0j * psi_e) / np.sqrt(2)
ket_0 = tf.constant(psi_g, dtype=tf.complex128)
bra_1 = tf.constant(psi_e.T, dtype=tf.complex128)
bra_ym = tf.constant(psi_ym.T, dtype=tf.complex128)


# TODO move fidelity experiments elsewhere
def evaluate_signals(pulse_values: list, opt_map: list):
    gates.set_parameters(pulse_values, opt_map)
    signal = gen.generate_signals(gates.instructions["X90p"])
    U = sim.propagation(signal)
    ket_actual = tf.matmul(U, ket_0)
    overlap = tf.matmul(bra_ym, ket_actual)
    return 1 - tf.cast(tf.math.conj(overlap) * overlap, tf.float64)


# Optimizer object
opt = Opt(data_path=logdir)
opt_map = [
    [('X90p', 'd1', 'gauss', 'amp')],
    [('X90p', 'd1', 'gauss', 'freq_offset')],
    [('X90p', 'd1', 'gauss', 'xy_angle')],
    [('X90p', 'd1', 'gauss', 'delta')]
]
opt.optimize_controls(
    controls=gates,
    opt_map=opt_map,
    opt='lbfgs',
    calib_name='openloop',
    eval_func=evaluate_signals
)
