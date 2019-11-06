"""Script to run orbit optimization."""

import pickle
import numpy as np
import tensorflow as tf
from scipy.linalg import expm as expm
import c3po.hamiltonians as hamiltonians
from c3po.simulator import Simulator as Sim
from c3po.optimizer import Optimizer as Opt
from c3po.experiment import Experiment as Exp
from c3po.utils import log_setup
from c3po.tf_utils import tf_matmul_list as tf_matmul_list
from c3po.tf_utils import tf_abs as tf_abs
from IBM_1q_chip import create_chip_model, create_generator, create_gates

logdir = log_setup("/tmp/c3logs/")

# System
qubit_freq = 5.1173e9 * 2 * np.pi
qubit_anhar = -315513734 * 2 * np.pi
qubit_lvls = 4
drive_ham = hamiltonians.x_drive
v_hz_conversion = 1e9 * 0.31
t_final = 15e-9
t1 = 30e-6
t2star = 25e-6
temp = 0.0  # i.e. don't thermalise but dissipate

# Define the ground state
qubit_g = np.zeros([qubit_lvls, 1])
qubit_g[0] = 1
ket_0 = tf.constant(qubit_g, tf.complex128)
bra_0 = tf.constant(qubit_g.T, tf.complex128)

# Simulation variables
sim_res = 60e9  # 600GHz
awg_res = 1.2e9  # 1.2GHz

# Create system
model = create_chip_model(
    qubit_freq,
    qubit_anhar,
    qubit_lvls,
    drive_ham,
    t1,
    t2star,
    temp
)
gen = create_generator(sim_res, awg_res, v_hz_conversion)
gen.devices['awg'].options = 'drag'
gates = create_gates(
    t_final,
    v_hz_conversion,
    qubit_freq,
    qubit_anhar,
    awg_res
)

exp = Exp(model, gen)
sim = Sim(exp, gates)
a_q = model.ann_opers[0]
sim.VZ = expm(1.0j * np.matmul(a_q.T.conj(), a_q) * qubit_freq * t_final)


def match_ORBIT(
    exp_params: list,
    exp_opt_map: list,
    gateset_values: list,
    gateset_opt_map: list,
    seq: list,
    fid: np.float64
):
    exp.set_parameters(exp_params, exp_opt_map)
    U = sim.evaluate_sequence(gateset_values, gateset_opt_map, seq)
    ket_actual = tf.matmul(U, ket_0)
    overlap = tf.matmul(bra_0, ket_actual)
    fid_sim = (1 - tf.cast(tf.linalg.adjoint(overlap) * overlap, tf.float64))
    diff = fid_sim - fid
    return diff


# IBM OTHER PARAMS
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

exp_opt_map = [
#    ('Q1', 'freq'),
#    ('Q1', 'anhar'),
#    ('Q1', 't1'),
#    ('Q1', 't2star'),
    ('v_to_hz', 'V_to_Hz')
]
# exp_opt_map = exp.list_parameters()

gateset_opt_map = [
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
with open('learn_from.pickle', 'rb+') as file:
    learn_from = pickle.load(file)
opt = Opt(data_path=logdir)
opt.gateset_opt_map = gateset_opt_map
opt.exp_opt_map = exp_opt_map
opt.random_samples = False
opt.batch_size = 14
opt.learn_from = learn_from
opt.learn_model(
    exp,
    eval_func=match_ORBIT
)


# gate_dict = sim.get_gates(learn_from[1895][0], gateset_opt_map)
# fid = learn_from[1895][2]
# seq = learn_from[1895][1]
# Us = []
# for gate in seq:
#     Us.append(gate_dict[gate])
# U = tf_matmul_list(Us)
# ket_actual = tf.matmul(U, ket_0)
# overlap = tf.matmul(bra_0, ket_actual)
# fid_sim = (1 - tf.cast(tf.linalg.adjoint(overlap) * overlap, tf.float64))
# print('overlap of final state with ground: ', overlap.numpy())
# print('1-|overlap| i.e. population in 1: ', float(fid_sim.numpy()))
# print('population in 1 in the experiment: ', fid)
#
# signal = gen.generate_signals(gates.instructions["Y90p"])
# y90p = sim.propagation(signal)
# sim.plot_dynamics(ket_0)
