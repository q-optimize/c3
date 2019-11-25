"""General clean OC script."""

import numpy as np
import tensorflow as tf
import c3po.hamiltonians as hamiltonians
from c3po.utils import log_setup
from scipy.linalg import expm
from c3po.component import Quantity as Qty
from c3po.simulator import Simulator as Sim
from c3po.optimizer import Optimizer as Opt
from c3po.experiment import Experiment as Exp
from c3po.tf_utils import tf_abs
from c3po.qt_utils import basis, xy_basis, perfect_gate
from single_qubit import create_chip_model, create_generator, create_gates

logdir = log_setup("/tmp/c3logs/")

# System
qubit_freq = Qty(
    value=5.1173e9 * 2 * np.pi,
    min=5.1e9 * 2 * np.pi,
    max=5.14e9 * 2 * np.pi,
    unit='rad'
)
qubit_anhar = Qty(
    value=-315.513734e6 * 2 * np.pi,
    min=-330e6 * 2 * np.pi,
    max=-300e6 * 2 * np.pi,
    unit='rad'
)
qubit_lvls = 4
drive_ham = hamiltonians.x_drive
v_hz_conversion = Qty(
    value=1,
    min=0.5,
    max=1.5,
    unit='rad/V'
)
t_final = Qty(
    value=10e-9,
    min=5e-9,
    max=15e-9,
    unit='s'
)
rise_time = Qty(
    value=0.1e-9,
    min=0.0e-9,
    max=0.2e-9,
    unit='s'
)

# Simulation variables
sim_res = 3e11
awg_res = 2.4e9

# Create system
model = create_chip_model(qubit_freq, qubit_anhar, qubit_lvls, drive_ham)
gen = create_generator(sim_res, awg_res, v_hz_conversion, logdir=logdir,
                       rise_time=rise_time)
gates = create_gates(t_final, v_hz_conversion, qubit_freq, qubit_anhar)

# gen.devices['awg'].options = 'drag'

# Simulation class and fidelity function
exp = Exp(model, gen)
sim = Sim(exp, gates)
a_q = model.ann_opers[0]
sim.VZ = expm(
    1.0j * np.matmul(a_q.T.conj(), a_q)
    * qubit_freq.get_value() * t_final.get_value()
    )

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
opt = Opt(data_path=logdir)
opt_map = [
    [('X90p', 'd1', 'gauss', 'amp')],
    [('X90p', 'd1', 'gauss', 'freq_offset')],
    # [('X90p', 'd1', 'gauss', 'xy_angle')],
    # [('X90p', 'd1', 'gauss', 'delta')]
]
opt.optimize_controls(
    sim=sim,
    opt_map=opt_map,
    opt='lbfgs',
    opt_name='openloop',
    fid_func=unitary_infid
)
