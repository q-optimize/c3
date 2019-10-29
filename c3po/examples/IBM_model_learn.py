"""Script to run orbit optimization."""

import pickle
import numpy as np
import tensorflow as tf
from scipy.linalg import expm as expm
import c3po.hamiltonians as hamiltonians
from c3po.simulator import Simulator as Sim
from c3po.optimizer import Optimizer as Opt
from IBM_1q_chip import create_chip_model, create_generator, create_gates

# System
qubit_freq = 5.1173e9 * 2 * np.pi
qubit_anhar = -3155137343 * 2 * np.pi
qubit_lvls = 6
drive_ham = hamiltonians.x_drive
v_hz_conversion = 1e9 * 0.40605
t_final = 15e-9

# Define the ground state
qubit_g = np.zeros([qubit_lvls, 1])
qubit_g[0] = 1
ket_0 = tf.constant(qubit_g, tf.complex128)
bra_0 = tf.constant(qubit_g.T, tf.complex128)

# Simulation variables
sim_res = 6e11  # 600GHz
awg_res = 1.2e9  # 1.2GHz

# Create system
model = create_chip_model(qubit_freq, qubit_anhar, qubit_lvls, drive_ham)
gen = create_generator(sim_res, awg_res, v_hz_conversion)
gates = create_gates(t_final, v_hz_conversion, qubit_freq, qubit_anhar)

opt = Opt()
sim = Sim(model, gen, gates)
a_q = model.ann_opers[0]
sim.VZ = expm(1.0j*a_q.T.conj()@a_q*qubit_freq*t_final)

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
    fid_sim = (1-tf.cast(tf.linalg.adjoint(overlap)*overlap, tf.float64))
    print('overlap of final state with ground ', overlap.numpy())
    print('1-|overlap| i.e. population in 1', fid_sim.numpy())
    print('population in 1 in the experiment', fid)
    diff = fid_sim - fid
    model_error = diff * diff
    return model_error

####### IBM OTHER PARAMS
# phase for gates = 0.0
# X90p xyangle = 0.0399 (radians)
# Y90p xyangle = 1.601
# Y90m xyangle = 4.7537
# X90m xyangle = 3.160
# width = 18 length in terms of number of AWG samples
# sigma = width/4 = 4.5 (in terms of number of AWG samples)
# 12 points in cloud
# 20 sequences per point
# 30 iterations

opt_map = [
    [('X90p', 'd1', 'gauss', 'amp'),
     ('Y90p', 'd1', 'gauss', 'amp'),
     ('X90m', 'd1', 'gauss', 'amp'),
     ('Y90m', 'd1', 'gauss', 'amp')],
    [('X90p', 'd1', 'gauss', 'freq_offset'),
     ('Y90p', 'd1', 'gauss', 'freq_offset'),
     ('X90m', 'd1', 'gauss', 'freq_offset'),
     ('Y90m', 'd1', 'gauss', 'freq_offset')],
    [('X90p', 'd1', 'gauss', 'delta'),
     ('Y90p', 'd1', 'gauss', 'delta'),
     ('X90m', 'd1', 'gauss', 'delta'),
     ('Y90m', 'd1', 'gauss', 'delta')],
]
opt.opt_map = opt_map
opt.random_samples = True
with open('/home/usersFWM/froy/Documents/PHD/other_code/IBMZ/learn_from.pickle', 'rb+') as file:
    learn_from = pickle.load(file)
# gen.devices['awg'].options = 'drag'
opt.learn_from = learn_from
opt.learn_model(
    model,
    eval_func=match_ORBIT
)

# gates.set_parameters(learn_from[976][0], opt_map)
# model_params, _ = model.get_values_bounds()
# signal = gen.generate_signals(gates.instructions["X90p"])
# U = sim.propagation(signal, model_params)
