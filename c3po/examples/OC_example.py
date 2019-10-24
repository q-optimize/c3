"""General clean OC script."""

import c3po.envelopes as envelopes
# import c3po.tf_utils as tf_utils
import c3po.component as component
import c3po.control as control
import c3po.generator as generator
import c3po.hamiltonians as hamiltonians
from c3po.model import Model as Mdl
from c3po.optimizer import Optimizer as Opt
from c3po.simulator import Simulator as Sim

import numpy as np
import tensorflow as tf

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
q1 = component.Qubit(
    name="Q1",
    desc="Qubit 1",
    comment="The one and only qubit in this chip",
    freq=qubit_freq,
    anhar=qubit_anhar,
    hilbert_dim=qubit_lvls
    )

drive = component.Drive(
    name="D1",
    desc="Drive 1",
    comment="Drive line 1 on qubit 1",
    connected=["Q1"],
    hamiltonian=hamiltonians.x_drive
    )

chip_elements = [q1, drive]
# TODO move mv2amp to the generator
simple_model = Mdl(chip_elements, mV_to_Amp)

# Devices and generator
lo = generator.LO(resolution=sim_res)
awg = generator.AWG(resolution=awg_res)
mixer = generator.Mixer()
devices = {
    "lo": lo,
    "awg": awg,
    "mixer": mixer
}
gen = generator.Generator(devices)

# Pulses and Instruction
# pwc_params = {
#     'inphase': np.ones(slice_num)*0.5*amp_limit,
#     'quadrature': np.ones(slice_num)*0.5*amp_limit
# }
# pwc_bounds = {
#     'inphase': [-amp_limit, amp_limit]*slice_num,
#     'quadrature': [-amp_limit, amp_limit]*slice_num
#     }
# pwc_env = component.Envelope(
#     name="pwc",
#     desc="PWC comp 1 of signal 1",
#     params=pwc_params,
#     bounds=pwc_bounds,
#     shape=envelopes.pwc
# )
gauss_params = {
        'amp': np.pi / mV_to_Amp,
        't_final': 8e-9,
        'xy_angle': 0.0,
        'freq_offset': 0e6 * 2 * np.pi
    }
gauss_bounds = {
        'amp': [0.01 * np.pi / mV_to_Amp, 1.5 * np.pi / mV_to_Amp],
        't_final': [7e-9, 12e-9],
        'xy_angle': [-1 * np.pi/2, 1 * np.pi/2],
        'freq_offset': [-100 * 1e6 * 2 * np.pi, 100 * 1e6 * 2 * np.pi]
    }
gauss_env = component.Envelope(
    name="gauss",
    desc="Gaussian comp 1 of signal 1",
    params=gauss_params,
    bounds=gauss_bounds,
    shape=envelopes.gaussian
)
carrier_parameters = {
    'freq': 5.95e9 * 2 * np.pi
}
carrier_bounds = {
    'freq': [5e9 * 2 * np.pi, 7e9 * 2 * np.pi]
}
carr = component.Carrier(
    name="carrier",
    desc="Frequency of the local oscillator",
    params=carrier_parameters,
    bounds=carrier_bounds
)
ctrl = control.Instruction(
    name="line1",
    t_start=0.0,
    t_end=t_final,
    comps=[carr, gauss_env]
)
ctrls = control.GateSet([ctrl])

# Simulation class and fidelity function
sim = Sim(simple_model, gen, ctrls)

qubit_g = np.zeros([qubit_lvls, 1])
qubit_g[0] = 1

qubit_e = np.zeros([qubit_lvls, 1])
qubit_e[1] = 1

ket_init = tf.constant(qubit_g, tf.complex128)
bra_goal = tf.constant(qubit_e.T, tf.complex128)


# TODO move fidelity experiments elsewhere
def evaluate_signals(pulse_values: list, opt_map: list):
    model_params = sim.model.params
    U = sim.propagation(pulse_values, opt_map, model_params)
    psi_actual = tf.matmul(U, ket_init)
    overlap = tf.matmul(bra_goal, psi_actual)
    return 1-tf.cast(tf.math.conj(overlap)*overlap, tf.float64)


# Optimizer object
opt = Opt()
# TODO modify use of opt_map in optimizer and in control
# opt_map = [
#     ('line1', 'pwc', 'inphase'),
#     ('line1', 'pwc', 'quadrature')
# ]
opt_map = [
    ('line1', 'gauss', 'amp'),
    ('line1', 'gauss', 'freq_offset')
]
opt.optimize_controls(
    controls=ctrls,
    opt_map=opt_map,
    opt='lbfgs',
    calib_name='openloop',
    eval_func=evaluate_signals
    )
