"""Script to run orbit optimization."""

import numpy as np
import tensorflow as tf
import c3po.hamiltonians as hamiltonians
from c3po.simulator import Simulator as Sim
from c3po.optimizer import Optimizer as Opt
from single_qubit import create_chip_model, create_generator, create_gates

# System
qubit_freq = 6e9 * 2 * np.pi
qubit_anhar = -100e6 * 2 * np.pi
qubit_lvls = 6
drive_ham = hamiltonians.x_drive
v_hz_conversion = 2e9 * np.pi
t_final = 10e-9

# Define the ground state
qubit_g = np.zeros([qubit_lvls, 1])
qubit_g[0] = 1
ket_0 = tf.constant(qubit_g, tf.complex128)
bra_0 = tf.constant(qubit_g.T, tf.complex128)

# Simulation variables
sim_res = 1e11
awg_res = 1e9  # 1.2GHz

# Create system
model = create_chip_model(qubit_freq, qubit_anhar, qubit_lvls, drive_ham)
gen = create_generator(sim_res, awg_res, v_hz_conversion)
gates = create_gates(t_final, v_hz_conversion, qubit_freq, qubit_anhar)

opt = Opt()
sim = Sim(model, gen, gates)

def match_ORBIT(
    model_values: list,
    gateset_values: list,
    opt_map: list,
    seq: list,
    fid: np.float64
):
    U = sim.evaluate_sequence(model_values, gateset_values, opt_map, seq)
    ket_actual = tf.matmul(U, ket_0)
    overlap = tf.matmul(bra_0, ket_actual)
    diff = (1-tf.cast(tf.linalg.adjoint(overlap)*overlap, tf.float64)) - fid
    model_error = diff * diff
    return model_error

opt_map = [
    [('X90p', 'd1', 'gauss', 'amp'),
     ('Y90p', 'd1', 'gauss', 'amp'),
     ('X90m', 'd1', 'gauss', 'amp'),
     ('Y90m', 'd1', 'gauss', 'amp')],
    [('X90p', 'd1', 'gauss', 'freq_offset'),
     ('Y90p', 'd1', 'gauss', 'freq_offset'),
     ('X90m', 'd1', 'gauss', 'freq_offset'),
     ('Y90m', 'd1', 'gauss', 'freq_offset')],
]
opt.opt_map = opt_map
opt.random_samples = True
opt.learn_from = [
    ([np.pi/v_hz_conversion, 1e6*2*np.pi], ['X90p', 'Y90m', 'Y90p', 'X90m'], 0.99),
    ([1.1*np.pi/v_hz_conversion, 1.1e6*2*np.pi], ['X90m', 'Y90m', 'Y90p', 'X90m', 'X90m', 'X90m'], 0.97),
    ([0.9*np.pi/v_hz_conversion, 0.9e6*2*np.pi], ['X90m', 'Y90m', 'Y90p', 'X90p', 'X90p', 'X90m'], 0.974)
]
opt.learn_model(
    model,
    eval_func=match_ORBIT
)
