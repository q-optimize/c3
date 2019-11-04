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

# System
qubit_freq = 5.1173e9 * 2 * np.pi
qubit_anhar = -3155137343 * 2 * np.pi
qubit_lvls = 4
drive_ham = hamiltonians.x_drive
v_hz_conversion = 1e9 * 0.31
t_final = 5e-09
t1 = 30e-6
t2star = 25e-6
temp = 70e-3  # i.e. don't thermalise but dissipate

# Define the ground state
qubit_g = np.zeros([qubit_lvls, 1])
qubit_g[0] = 1
ket_0 = tf.constant(qubit_g, tf.complex128)
bra_0 = tf.constant(qubit_g.T, tf.complex128)

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
# gen.devices['awg'].options = 'drag'
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

gateset_opt_map = [
            [('X90p', 'd1', 'gauss', 'amp')],
            [('X90p', 'd1', 'gauss', 'freq_offset')],
            # [('X90p', 'd1', 'gauss', 'delta')]
]

# Define states & unitaries
# TODO create basic unit vectors with a function call or an import
psi_g = np.zeros([qubit_lvls, 1])
psi_g[0] = 1
psi_e = np.zeros([qubit_lvls, 1])
psi_e[1] = 1
ket_0 = tf.constant(psi_g, dtype=tf.complex128)
bra_1 = tf.constant(psi_e.T, dtype=tf.complex128)
psi_ym = (psi_g - 1.0j * psi_e) / np.sqrt(2)
bra_ym = tf.constant(psi_ym.T, dtype=tf.complex128)
psi_yp = (psi_g + 1.0j * psi_e) / np.sqrt(2)
bra_yp = tf.constant(psi_yp.T, dtype=tf.complex128)
psi_xm = (psi_g - psi_e) / np.sqrt(2)
bra_xm = tf.constant(psi_xm.T, dtype=tf.complex128)
psi_xp = (psi_g + psi_e) / np.sqrt(2)
bra_xp = tf.constant(psi_xp.T, dtype=tf.complex128)

# pulse_params = [1.49775076e+00, -1.00676867e+07, -4.59827744e-01]
# pulse_params = [1.70349896e+00,  3.08792384e+07, -2.53171726e-01]
pulse_params = [1.68732807e+00, -3.85382920e+06]
gates.set_parameters(pulse_params, gateset_opt_map)
signal = gen.generate_signals(gates.instructions["X90p"])
U = sim.propagation(signal)
ket_actual = tf.matmul(U, ket_0)
overlap = tf.matmul(bra_yp, ket_actual)
overlap_abs = tf.cast(tf.math.conj(overlap)*overlap, tf.float64)
infid = 1 - overlap_abs
print('overlap')
print(overlap)
print('overlap fidelity')
print(infid)
gen.devices['awg'].plot_IQ_components()
sim.plot_dynamics(ket_0)
