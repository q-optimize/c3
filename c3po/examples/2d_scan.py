"""Looking at the best we can do for unitary overlap."""

import pickle
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
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
pulse_type = 'gauss'
qubit_freq = 5.1173e9 * 2 * np.pi
qubit_anhar = -315.513734e6 * 2 * np.pi
qubit_lvls = 4
drive_ham = hamiltonians.x_drive
v_hz_conversion = 1
t_final = 5e-09

# Simulation variables
sim_res = 3e11  # 300GHz
awg_res = 1.2e9  # 1.2GHz

# Create system
model = create_chip_model(qubit_freq, qubit_anhar, qubit_lvls, drive_ham)
gen = create_generator(sim_res, awg_res, v_hz_conversion, logdir=logdir)
gates = create_gates(t_final, v_hz_conversion, qubit_freq, qubit_anhar)

exp = Exp(model, gen)
sim = Sim(exp, gates)
a_q = model.ann_opers[0]
sim.VZ = expm(1.0j * np.matmul(a_q.T.conj(), a_q) * qubit_freq * t_final)

# Define states & unitaries
ket_0 = tf.constant(basis(qubit_lvls, 0), dtype=tf.complex128)
bra_yp = tf.constant(xy_basis(qubit_lvls, 'yp').T, dtype=tf.complex128)
ket_xp = tf.constant(xy_basis(qubit_lvls, 'xp'), dtype=tf.complex128)
bra_xp = tf.constant(xy_basis(qubit_lvls, 'xp').T, dtype=tf.complex128)
X90p = tf.constant(perfect_gate(qubit_lvls, 'X90p'), dtype=tf.complex128)


gateset_opt_map = [
    [('X90p', 'd1', 'gauss', 'amp')],
    [('X90p', 'd1', 'gauss', 'freq_offset')]
]

amp_num = 51
freq_num = 11
amps = np.linspace(1/4, 3/4, num=amp_num) * np.pi
freqs = np.linspace(-1, 1, num=freq_num) * 1e4
psi0_overlaps = np.zeros([amp_num, freq_num])
psixp_overlaps = np.zeros([amp_num, freq_num])
unit_fids = np.zeros([amp_num, freq_num])
for amp_indx in range(amp_num):
    for freq_indx in range(freq_num):
        print('iteration:', amp_indx*freq_num+freq_indx)
        print('indeces', amp_indx, freq_indx)
        amp = amps[amp_indx]
        freq = freqs[freq_indx]
        pulse_params = [amp, freq]
        gates.set_parameters(pulse_params, gateset_opt_map)
        signal = gen.generate_signals(gates.instructions["X90p"])
        U = sim.propagation(signal)
        psi0_overlap = tf_abs(tf.matmul(bra_yp,  tf.matmul(U, ket_0)))
        psixp_overlap = tf_abs(tf.matmul(bra_xp,  tf.matmul(U, ket_xp)))
        unit_fid = tf_abs(
                    tf.linalg.trace(
                        tf.matmul(U, tf.linalg.adjoint(X90p))
                        ) / 2
                    )**2
        print('psi0_overlap', psi0_overlap)
        print('psixp_overlap', psixp_overlap)
        print('unit_fid', unit_fid)
        psi0_overlaps[amp_indx, freq_indx] = psi0_overlap
        psixp_overlaps[amp_indx, freq_indx] = psixp_overlap
        unit_fids[amp_indx, freq_indx] = unit_fid

    # save before fucking up the plot
    with open('2dplot.pickle', 'wb') as file:
        pickle.dump(amps, file, protocol=pickle.HIGHEST_PROTOCOL)
        pickle.dump(freqs, file, protocol=pickle.HIGHEST_PROTOCOL)
        pickle.dump(psi0_overlaps, file, protocol=pickle.HIGHEST_PROTOCOL)
        pickle.dump(psixp_overlaps, file, protocol=pickle.HIGHEST_PROTOCOL)
        pickle.dump(unit_fids, file, protocol=pickle.HIGHEST_PROTOCOL)


# save before fucking up the plot
with open('2dplot.pickle', 'rb') as file:
    amps = pickle.load(file)
    freqs = pickle.load(file)
    psi0_overlaps = pickle.load(file)
    psixp_overlaps = pickle.load(file)
    unit_fids = pickle.load(file)

# plot 2d fidelities
plt.contourf(amps, freqs, unit_fids.T)
plt.colorbar()
