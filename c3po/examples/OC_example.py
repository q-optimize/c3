"""General clean OC script."""

import numpy as np
import tensorflow as tf
import c3po.hamiltonians as hamiltonians
from c3po.simulator import Simulator as Sim
from c3po.optimizer import Optimizer as Opt
from c3po.experiment import Experiment as Exp
from c3po.tf_utils import tf_abs, tf_ave
from c3po.qt_utils import basis, xy_basis, perfect_gate
from single_qubit import create_chip_model, create_generator, create_gates

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
gen = create_generator(sim_res, awg_res, v_hz_conversion)
gates = create_gates(t_final, v_hz_conversion, qubit_freq, qubit_anhar)

# Simulation class and fidelity function
exp = Exp(model, gen)
sim = Sim(exp, gates)

# Define states
# Define states & unitaries
ket_0 = tf.constant(basis(qubit_lvls, 0), dtype=tf.complex128)
bra_2 = tf.constant(basis(qubit_lvls, 2).T, dtype=tf.complex128)
bra_yp = tf.constant(xy_basis(qubit_lvls, 'yp').T, dtype=tf.complex128)
X90p = tf.constant(perfect_gate(qubit_lvls, 'X90p'), dtype=tf.complex128)


# TODO move fidelity experiments elsewhere
def state_transfer_infid(U_dict: dict):
    U = U_dict['X90p']
    ket_actual = tf.matmul(U, ket_0)
    overlap = tf_abs(tf.matmul(bra_yp, ket_actual))
    infid = 1 - overlap
    return infid

def unitary_infid(U_dict: dict):
    U = U_dict['X90p']
    unit_fid = tf_abs(
                tf.linalg.trace(
                    tf.matmul(U, tf.linalg.adjoint(X90p))
                    ) / 2
                )**2
    infid = 1 - unit_fid
    return infid

def pop_leak(U_dict: dict):
    U = U_dict['X90p']
    ket_actual = tf.matmul(U, ket_0)
    overlap = tf_abs(tf.matmul(bra_2, ket_actual))
    return overlap

# Optimizer object
opt = Opt()
opt_map = [
    [('X90p', 'd1', 'gauss', 'amp')],
    [('X90p', 'd1', 'gauss', 'freq_offset')],
    [('X90p', 'd1', 'gauss', 'xy_angle')]
]
opt.optimize_controls(
    sim=sim,
    opt_map=opt_map,
    opt='lbfgs',
    optim_name='openloop',
    fid_func=state_transfer_infid
    )
