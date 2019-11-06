"""Script to run orbit optimization."""

import numpy as np
import tensorflow as tf
from scipy.linalg import expm as expm
import c3po.hamiltonians as hamiltonians
from c3po.simulator import Simulator as Sim
# from c3po.optimizer import Optimizer as Opt
from c3po.experiment import Experiment as Exp
# from c3po.tf_utils import tf_matmul_list as tf_matmul_list
from IBM_1q_chip import create_chip_model, create_generator, create_gates
from c3po.tf_utils import tf_abs
from c3po.qt_utils import basis, xy_basis, perfect_gate

# System
pulse_type = 'drag'  # 'gauss' 'drag' 'pwc'
qubit_freq = 5.1173e9 * 2 * np.pi
qubit_anhar = -315.513734e6 * 2 * np.pi / 3
qubit_lvls = 4
drive_ham = hamiltonians.x_drive
v_hz_conversion = 1e9 * 0.31
t_final = 2.5e-09
t1 = None,  # 30e-6
t2star = None,  # 30e-6
temp = None,  # 70e-3

# Simulation variables
sim_res = 3e11  # 600GHz
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
if pulse_type == 'drag':
    gen.devices['awg'].options = 'drag'
elif pulse_type == 'pwc':
    gen.devices['awg'].options = 'pwc'
gates = create_gates(t_final,
                     qubit_freq,
                     qubit_anhar,
                     amp=1.3,
                     IBM_angles=False
                     )

exp = Exp(model, gen)
sim = Sim(exp, gates)
a_q = model.ann_opers[0]
sim.VZ = expm(1.0j * np.matmul(a_q.T.conj(), a_q) * qubit_freq * t_final)

if pulse_type == 'gauss':
    gateset_opt_map = [
        [('X90p', 'd1', 'gauss', 'amp')],
        [('X90p', 'd1', 'gauss', 'freq_offset')],
        [('X90p', 'd1', 'gauss', 'xy_angle')]
    ]
if pulse_type == 'drag':
    gateset_opt_map = [
                [('X90p', 'd1', 'gauss', 'amp')],
                [('X90p', 'd1', 'gauss', 'freq_offset')],
                [('X90p', 'd1', 'gauss', 'xy_angle')],
                [('X90p', 'd1', 'gauss', 'delta')]
    ]

# Define states & unitaries
ket_0 = tf.constant(basis(qubit_lvls, 0), dtype=tf.complex128)
bra_0 = tf.constant(basis(qubit_lvls, 0).T, dtype=tf.complex128)
bra_yp = tf.constant(xy_basis(qubit_lvls, 'yp').T, dtype=tf.complex128)
X90p = tf.constant(perfect_gate(qubit_lvls, 'X90p'), dtype=tf.complex128)

if pulse_type == 'gauss':
    pulse_params = [3.29492373e+00,  9.99088882e+07, -1.27116729e-01]
if pulse_type == 'drag':
    pulse_params = [3.29078791e+00, -8.00675235e+04, -1.91233731e-03, -7.00661416e-02]
gates.set_parameters(pulse_params, gateset_opt_map)
signal = gen.generate_signals(gates.instructions["X90p"])
U = sim.propagation(signal)
ket_actual = tf.matmul(U, ket_0)
overlap = tf_abs(tf.matmul(bra_yp, ket_actual))
ov_infid = 1 - overlap
unit_fid = tf_abs(
            tf.linalg.trace(
                tf.matmul(U, tf.linalg.adjoint(X90p))
                ) / 2
            )**2
un_infid = 1 - unit_fid
print('overlap error')
print(ov_infid)
print('unitary error')
print(un_infid)
sim.plot_dynamics(ket_0)
