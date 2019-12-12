"""
Synthetic C3 model learning

Construct two models, a 'wrong' one and a 'correct' one. First construct
open loop pulses with the wrong model then calibrate them against the real
model, simulating an experiment. Finally try to match the wrong model to the
calibration data and recover the real one.
"""

import json
import numpy as np
import tensorflow as tf
import c3po.hamiltonians as hamiltonians
from c3po.utils import log_setup
from c3po.component import Quantity as Qty
from c3po.simulator import Simulator as Sim
from c3po.optimizer import Optimizer as Opt
from c3po.experiment import Experiment as Exp
from c3po.tf_utils import tf_abs
from c3po.qt_utils import basis, xy_basis, perfect_gate
from single_qubit import create_chip_model, create_generator, create_gates
from c3po.tf_utils import tf_limit_gpu_memory

tf_limit_gpu_memory(100)
run_name = "synthetic_rnd_smpl"
logdir = log_setup("/tmp/c3logs/", run_name=run_name)

with tf.device('/CPU:0'):
    # System
    qubit_freq = Qty(
        value=5.1173e9 * 2 * np.pi,
        min=5.1e9 * 2 * np.pi,
        max=5.14e9 * 2 * np.pi,
        unit='Hz 2pi'
    )
    qubit_anhar = Qty(
        value=-315.513734e6 * 2 * np.pi,
        min=-330e6 * 2 * np.pi,
        max=-300e6 * 2 * np.pi,
        unit='Hz 2pi'
    )
    qubit_lvls = 4
    drive_ham = hamiltonians.x_drive
    v_hz_conversion = Qty(
        value=1,
        min=0.9,
        max=1.1,
        unit='rad/V'
    )

    qubit_freq_wrong = Qty(
        value=5.1173e9 * 2 * np.pi,
        min=5.1e9 * 2 * np.pi,
        max=5.14e9 * 2 * np.pi,
        unit='Hz 2pi'
    )
    qubit_anhar_wrong = Qty(
        value=-315.513734e6 * 2 * np.pi,
        min=-330e6 * 2 * np.pi,
        max=-300e6 * 2 * np.pi,
        unit='Hz 2pi'
    )
    qubit_lvls = 4
    drive_ham = hamiltonians.x_drive
    v_hz_conversion_wrong = Qty(
        value=0.95,
        min=0.9,
        max=1.1,
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

    carrier_freq = Qty(
        value=5.1e9 * 2 * np.pi,
        min=5e9 * 2 * np.pi,
        max=5.5e9 * 2 * np.pi,
        unit='Hz 2pi'
    )

    freq_offset = Qty(
        value=0e6 * 2 * np.pi,
        min=-250 * 1e6 * 2 * np.pi,
        max=250 * 1e6 * 2 * np.pi,
        unit='Hz 2pi'
    )

    # Define the ground state
    qubit_g = np.zeros([qubit_lvls, 1])
    qubit_g[0] = 1
    ket_0 = tf.constant(qubit_g, tf.complex128)
    bra_0 = tf.constant(qubit_g.T, tf.complex128)

    # Simulation variables
    sim_res = 60e9
    awg_res = 1.2e9

    # Create system
    model_wrong = create_chip_model(
        qubit_freq_wrong, qubit_anhar_wrong, qubit_lvls, drive_ham
    )
    model_right = create_chip_model(
        qubit_freq, qubit_anhar, qubit_lvls, drive_ham
    )
    gen_wrong = create_generator(
        sim_res, awg_res, v_hz_conversion_wrong, logdir=logdir,
        rise_time=rise_time
    )
    # gen_wrong.devices['awg'].options = 'drag'
    gen_right = create_generator(
        sim_res, awg_res, v_hz_conversion, logdir=logdir, rise_time=rise_time
    )
    # gen_right.devices['awg'].options = 'drag'
    gates = create_gates(
        t_final=t_final,
        v_hz_conversion=v_hz_conversion_wrong,
        qubit_freq=qubit_freq_wrong,
        qubit_anhar=qubit_anhar_wrong,
        all_gates=False,
        freq_offset=freq_offset,
        carrier_freq=carrier_freq
    )


    # gen.devices['awg'].options = 'drag'

    # Simulation class and fidelity function
    exp_wrong = Exp(model_wrong, gen_wrong)
    sim_wrong = Sim(exp_wrong, gates)
    sim_wrong.use_VZ = True

    exp_right = Exp(model_right, gen_right)
    sim_right = Sim(exp_right, gates)
    sim_right.use_VZ = True

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
            tf.linalg.trace(tf.matmul(U, tf.linalg.adjoint(X90p))) / 2
        )**2
        infid = 1 - unit_fid
        return infid

    def pop_leak(U_dict: dict):
        U = U_dict['X90p']
        ket_actual = tf.matmul(U, ket_0)
        overlap = tf_abs(tf.matmul(bra_2, ket_actual))
        return overlap

    def match_calib(
        U_dict,
        seq: list,
        fid: np.float64
    ):
        fid_sim = unitary_infid(U_dict)
        diff = fid_sim - fid
        return diff

    # Optimizer object
    opt_map = [
        [('X90p', 'd1', 'gauss', 'amp')],
        [('X90p', 'd1', 'gauss', 'freq_offset')],
        [('X90p', 'd1', 'gauss', 'xy_angle')],
        # [('X90p', 'd1', 'gauss', 'delta')]
    ]

    exp_opt_map = [
        ('Q1', 'freq'),
        # ('Q1', 'anhar'),
        # ('Q1', 't1'),
        # ('Q1', 't2star'),
        ('v_to_hz', 'V_to_Hz'),
        # ('resp', 'rise_time')
    ]

    def c3_openloop():
        opt = Opt(data_path=logdir)
        opt.optimize_controls(
            sim=sim_wrong,
            opt_map=opt_map,
            opt='lbfgs',
            opt_name='openloop',
            fid_func=unitary_infid
        )

    def c3_calibration(noise_level=0):
        opt = Opt(data_path=logdir)
        opt.noise_level = noise_level
        opt.optimize_controls(
            sim=sim_right,
            opt_map=opt_map,
            opt='cmaes',
            # settings={},
            settings={'ftarget': 1e-4},
            opt_name='calibration',
            fid_func=unitary_infid
        )

    def c3_learn_model(logfilename, sampling='even', batch_size=10):
        learn_from = []
        with open(logfilename, "r") as calibration_log:
            log = calibration_log.readlines()
        for line in log:
            if line[0] == "{":
                line_dict = json.loads(line)
                learn_from.append(
                    [line_dict['params'], [[['X90p'], line_dict['goal']]]]
                )
        opt = Opt(data_path=logdir)
        opt.gateset_opt_map = opt_map
        opt.opt_map = exp_opt_map
        opt.sampling = sampling
        opt.batch_size = batch_size
        opt.learn_from = learn_from
        opt.sim = sim_wrong
        settings = {'ftol': 1e-12}
        opt.learn_model(
            exp_wrong,
            sim_wrong,
            eval_func=match_calib,
            settings=settings
        )


# Run the stuff
    c3_openloop()
    c3_calibration(noise_level=0)
    logfilename = logdir + "calibration.log"
    #  sampling = 'from_end'  'even', 'random', 'from_start'
    c3_learn_model(logfilename, sampling='random', batch_size=10)
#

# c3_openloop()
# c3_calibration(noise_level=0)
#
# import matplotlib.pyplot as plt
# with tf.device("/CPU:0"):
#     learn_from = []
#     with open("/tmp/c3logs/recent/calibration.log", "r") as calibration_log:
#         log = calibration_log.readlines()
#     for line in log:
#         if line[0] == "{":
#             line_dict = json.loads(line)
#             learn_from.append(
#                 [line_dict['params'], ['X90p'], line_dict['goal']]
#             )
#
#     xrange = np.linspace(-0.14, -0.13, 21)
#     errs = []
#     for x in xrange:
#         print(f"Doing {x}")
#         n = int(len(learn_from) / 20)
#         measurements = learn_from[::n]
#         diff = []
#         for mes in measurements:
#             diff.append(
#                 match_calib(
#                     [x], exp_opt_map, mes[0], opt_map, mes[1], mes[2]
#                 )
#             )
#         rms = np.sqrt(np.mean(np.square(diff)))
#         print(f"RMS: {rms}")
#         errs.append(rms)
#
# plt.plot(xrange, errs)
