"""Testing if analytical drag helps."""

import numpy as np
import tensorflow as tf
from scipy.linalg import expm as expm
import c3po.hamiltonians as hamiltonians
from c3po.simulator import Simulator as Sim
from c3po.experiment import Experiment as Exp
from single_qubit import create_chip_model, create_generator, create_gates
from c3po.utils import log_setup
from c3po.tf_utils import tf_abs
from c3po.qt_utils import basis, xy_basis, perfect_gate

logdir = log_setup("/tmp/c3logs/")

# System
pulse_type = 'gauss'  # 'gauss' 'drag' 'pwc'
qubit_freq = 5.1173e9 * 2 * np.pi
qubit_anhar = -315.513734e6 * 2 * np.pi
qubit_lvls = 4
drive_ham = hamiltonians.x_drive
v_hz_conversion = 1
t_final = 5e-09

# Simulation variables
sim_res = 3e11
awg_res = 1.2e9

# Create system
model = create_chip_model(qubit_freq, qubit_anhar, qubit_lvls, drive_ham)
gen = create_generator(sim_res, awg_res, v_hz_conversion, logdir=logdir)
if pulse_type == 'drag':
    gen.devices['awg'].options = 'drag'
gates = create_gates(t_final, v_hz_conversion, qubit_freq, qubit_anhar)

exp = Exp(model, gen)
sim = Sim(exp, gates)
a_q = model.ann_opers[0]
sim.VZ = expm(1.0j * np.matmul(a_q.T.conj(), a_q) * qubit_freq * t_final)

# Define states & unitaries
ket_0 = tf.constant(basis(qubit_lvls, 0), dtype=tf.complex128)
bra_0 = tf.constant(basis(qubit_lvls, 0).T, dtype=tf.complex128)
bra_yp = tf.constant(xy_basis(qubit_lvls, 'yp').T, dtype=tf.complex128)
X90p = tf.constant(perfect_gate(qubit_lvls, 'X90p'), dtype=tf.complex128)

if pulse_type == 'gauss':
    gateset_opt_map = [
        [('X90p', 'd1', 'gauss', 'amp')],
        [('X90p', 'd1', 'gauss', 'freq_offset')]
    ]
# if pulse_type == 'drag':
#     gateset_opt_map = [
#                 [('X90p', 'd1', 'gauss', 'amp')],
#                 [('X90p', 'd1', 'gauss', 'freq_offset')],
#                 [('X90p', 'd1', 'gauss', 'xy_angle')],
#                 [('X90p', 'd1', 'gauss', 'delta')]
#     ]
# if pulse_type == 'gauss':
#     pulse_params = [7.44570979e-01, -1.70625351e+08]
# if pulse_type == 'drag':
#     pulse_params = []
# gates.set_parameters(pulse_params, gateset_opt_map)

current_params, _ = gates.get_parameters(gateset_opt_map)
U_dict = sim.get_gates(current_params, gateset_opt_map)
U = U_dict['X90p']
unit_fid = tf_abs(
            tf.linalg.trace(
                tf.matmul(U, tf.linalg.adjoint(X90p))
                ) / 2
            )**2
infid = 1 - unit_fid
ket_actual = tf.matmul(U, ket_0)
overlap = tf_abs(tf.matmul(bra_yp, ket_actual))
print('overlap error')
print(float((1 - overlap).numpy()))
print('unitary error')
print(float(infid.numpy()))
sim.plot_dynamics(ket_0)
