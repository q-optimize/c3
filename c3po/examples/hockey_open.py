"""Script to get optimization data."""

import os
import pickle
from datetime import datetime

import numpy as np
import tensorflow as tf
from c3po.utils import log_setup
from c3po.tf_utils import tf_abs, tf_ave, tf_super
from c3po.qt_utils import basis, xy_basis, perfect_gate, single_length_RB

from scipy.linalg import expm
import c3po.hamiltonians as hamiltonians
from c3po.simulator import Simulator as Sim
from c3po.optimizer import Optimizer as Opt
from c3po.experiment import Experiment as Exp
from single_qubit import create_chip_model, create_generator, create_gates, create_pwc_gates

# Script parameters
lindbladian = True
VZ = True
IBM_angles = False
search_fid = 'unit'  # 'state' 'unit' 'orbit'
pulse_type = 'pwc'  # 'gauss' 'drag' 'pwc'
sim_res = 3e11
awg_res = 2.4e9
awg_sample = 1 / awg_res
sample_numbers = np.arange(4, 20, 1)
logdir = log_setup("/tmp/c3logs/")
amp_limit = 3e9

# System
qubit_freq = 5.1173e9 * 2 * np.pi
qubit_anhar = -315.513734e6 * 2 * np.pi
qubit_lvls = 4
drive_ham = hamiltonians.x_drive
v_hz_conversion = 1
t1 = 30e-6
t2star = 25e-6
temp = 70e-3

# File and directory names name
base_dir = '/home/usersFWM/froy/Documents/PHD/'
dir_name = 'hockey_open/' + datetime.now().strftime('%Y-%m-%d_%H:%M') + '/'
specs_str = '{}_N{}'.format(pulse_type, qubit_lvls)
datafile = specs_str + '.out'
savefile = specs_str + '.pickle'
newpath = base_dir + dir_name
if not os.path.exists(newpath):
    os.makedirs(newpath)
os.system('cp hockey_open.py {}config.py'.format(newpath))
if pulse_type == 'pwc':
    with open('best_drag_awg.pickle', 'rb') as file:
        awg_values = pickle.load(file)

# Define states & unitaries
ket_0 = tf.constant(basis(qubit_lvls, 0), dtype=tf.complex128)
bra_0 = tf.constant(basis(qubit_lvls, 0).T, dtype=tf.complex128)
bra_yp = tf.constant(xy_basis(qubit_lvls, 'yp').T, dtype=tf.complex128)
X90p = tf.constant(perfect_gate(qubit_lvls, 'X90p'), dtype=tf.complex128)
if lindbladian:
    X90p = tf_super(X90p)

# Iter over different length pulses for CREATING DATA
overlap_inf = []
best_params = []
times = []
indx = 0
for sample_num in sample_numbers:
    print(": ", sample_num)
    # Update experiment time
    t_final = sample_num*awg_sample

    # Create system
    if not lindbladian:
        model = create_chip_model(qubit_freq, qubit_anhar, qubit_lvls, drive_ham)
    elif lindbladian:
        model = create_chip_model(qubit_freq, qubit_anhar, qubit_lvls, drive_ham,
                                  t1=t1, t2star=t2star, temp=temp)

    gen = create_generator(sim_res, awg_res, v_hz_conversion, logdir=logdir)

    if pulse_type == 'drag':
        gen.devices['awg'].options = 'drag'
    elif pulse_type == 'pwc':
        gen.devices['awg'].options = 'pwc'

    if not pulse_type == 'pwc':
        gates = create_gates(t_final, v_hz_conversion, qubit_freq, qubit_anhar)
    else:
        iq = awg_values[indx]
        indx = indx + 1
        inphase = iq['inphase']
        quadrature = iq['quadrature']
        gates = create_pwc_gates(t_final, qubit_freq, inphase, quadrature, amp_limit)
    exp = Exp(model, gen)
    sim = Sim(exp, gates)
    sim.lindbladian = lindbladian
    if VZ:
        a_q = model.ann_opers[0]
        sim.VZ = expm(1.0j * np.matmul(a_q.T.conj(), a_q) * qubit_freq * t_final)

    # optimizer
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
                    [('X90p', 'd1', 'gauss', 'delta')],
                    [('X90p', 'd1', 'gauss', 'xy_angle')]
        ]
    if pulse_type == 'pwc':
        gateset_opt_map = [
                    [('X90p', 'd1', 'pwc', 'inphase')],
                    [('X90p', 'd1', 'pwc', 'quadrature')],
        ]

    def state_transfer_infid(U_dict: dict):
        U = U_dict['X90p']
        ket_actual = tf.matmul(U, ket_0)
        overlap = tf_abs(tf.matmul(bra_yp, ket_actual))
        infid = 1 - overlap
        return infid

    def unitary_infid(U_dict: dict):
        U = U_dict['X90p']
        if not lindbladian:
            unit_fid = tf_abs(
                        tf.linalg.trace(
                            tf.matmul(U, tf.linalg.adjoint(X90p))
                            ) / 2
                        )**2
        elif lindbladian:
            unit_fid = tf_abs(
                        tf.sqrt(
                            tf.linalg.trace(
                                tf.matmul(U, tf.linalg.adjoint(X90p))
                                )
                            ) / 2
                        )**2
        infid = 1 - unit_fid
        return infid

    def orbit_infid(U_dict: dict):
        seqs = single_length_RB(RB_number=25, RB_length=20)
        U_seqs = sim.evaluate_sequences(U_dict, seqs)
        infids = []
        for U in U_seqs:
            ket_actual = tf.matmul(U, ket_0)
            overlap = tf_abs(tf.matmul(bra_0, ket_actual))
            infids.append(1 - overlap)
        return tf_ave(infids)

    opt = Opt(data_path=logdir)
    if search_fid == 'state':
        fid_func = state_transfer_infid
    elif search_fid == 'unit':
        fid_func = unitary_infid
    elif search_fid == 'orbit':
        fid_func = orbit_infid

    opt.optimize_controls(
        sim=sim,
        opt_map=gateset_opt_map,
        opt='lbfgs',
        opt_name='openloop',
        fid_func=fid_func
        )

    # store results
    overlap_inf.append(opt.results['openloop'].fun)
    best_params.append(opt.results['openloop'].x)
    times.append(t_final)
    print('times :', times)
    print('infid :', overlap_inf)
    # Save data
    with open('{}{}'.format(newpath, savefile), 'wb') as file:
        pickle.dump(overlap_inf, file, protocol=pickle.HIGHEST_PROTOCOL)
        pickle.dump(times, file, protocol=pickle.HIGHEST_PROTOCOL)
        pickle.dump(best_params, file, protocol=pickle.HIGHEST_PROTOCOL)
