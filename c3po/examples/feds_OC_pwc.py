from c3po.envelopes import *
from c3po.tf_utils import *
from c3po.component import *
from c3po.control import *
from c3po.generator import *
from c3po.hamiltonians import *
from c3po.model import Model as Mdl
from c3po.optimizer import Optimizer as Opt
from c3po.simulator import Simulator as Sim

import copy
import pickle
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

# System
qubit_freq = 6e9 * 2 * np.pi
qubit_anhar = -100e6 * 2 * np.pi
qubit_lvls = 6
mV_to_Amp = 2e9*np.pi

# For pwc calculate the number of slices beforehand:
awg_res = 1.2e9  # 1.2GHz
t_final = 8e-9  # 8ns
slice_num = int(t_final * awg_res)
amp_limit = 300e-3  # 300mV

# Simulation variables
sim_res = 5e10  # 50GHz

# Chip and model
q1 = Qubit(
    name="Q1",
    desc="Qubit 1",
    comment="The one and only qubit in this chip",
    freq=qubit_freq,
    anhar=qubit_anhar,
    hilbert_dim=qubit_lvls
    )

drive = Drive(
    name="D1",
    desc="Drive 1",
    comment="Drive line 1 on qubit 1",
    connected=[q1.name],
    hamiltonian=x_drive
    )

chip_elements = [q1, drive]
simple_model = Mdl(chip_elements, mV_to_Amp)

# Devices and generator
# TODO clean device classes
awg = AWG()
mixer = Mixer()

devices = {
    "awg": awg,
    "mixer": mixer
}

resolutions = {
    "awg": awg_res,
    "sim": sim_res
}

# TODO clean and automate generator class
gen = Generator()
gen.devices = devices
gen.resolutions = resolutions

# Pulses and Control
pwc_params = {
    'Inphase': np.ones(slice_num)*0.5*amp_limit,
    'Quadrature': np.ones(slice_num)*0.5*amp_limit
}

pwc_bounds = {
    'Inphase': [-amp_limit, amp_limit]*slice_num,
    'Quadrature': [-amp_limit, amp_limit]*slice_num
    }

carrier_parameters = {
    'freq': 5.95e9 * 2 * np.pi
}

carrier_bounds = {
    'freq': [5e9 * 2 * np.pi, 7e9 * 2 * np.pi]
}

env1 = Envelope(
    name="pwc",
    desc="PWC comp 1 of signal 1",
    params=pwc_params,
    bounds=pwc_bounds,
    shape=pwc
)

carr = Carrier(
    name="carrier",
    desc="Frequency of the local oscillator",
    params=carrier_parameters,
    bounds=carrier_bounds
)

# TODO clean control and controlset
ctrl = Control()
ctrl.name = "line1"
ctrl.t_start = 0.0
ctrl.t_end = t_final
ctrl.comps = [carr, env1]
ctrls = ControlSet([ctrl])

# Simulation class and fidelity function
# TODO clean simulation class
sim = Sim(simple_model, gen, ctrls)

qubit_g = np.zeros([qubit_lvls, 1])
qubit_g[0] = 1

qubit_e = np.zeros([qubit_lvls, 1])
qubit_e[1] = 1

ket_init = tf.constant(qubit_g, tf.complex128)
bra_goal = tf.constant(qubit_e.T, tf.complex128)


# TODO move fidelity experiments elsewhere
def evaluate_signals(pulse_params, opt_params):
    model_params = sim.model.params
    U = sim.propagation(pulse_params, opt_params, model_params)
    psi_actual = tf.matmul(U, ket_init)
    overlap = tf.matmul(bra_goal, psi_actual)
    return 1-tf.cast(tf.math.conj(overlap)*overlap, tf.float64)


# Optimizer object
opt = Opt()
# TODO modify use of opt_map in optimizer and in control
opt_map = {
    'Inphase': [('line1', 'env1')],
    'Quadrature': [('line1', 'env1')]
}

opt.optimize_controls(
    controls=ctrls,
    opt_map=opt_map,
    opt='lbfgs',
    calib_name='openloop',
    eval_func=evaluate_signals
    )
