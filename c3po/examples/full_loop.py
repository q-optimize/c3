import time
import uuid
import copy

from os import system

import numpy as np
import qutip as qt
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow.python import debug as tf_debug

from c3po.main.model import Model as mdl

from c3po.cobj.component import *

from c3po.utils.tf_utils import *
from c3po.control.envelopes import *

from c3po.cobj.component import ControlComponent as CtrlComp
from c3po.cobj.group import ComponentGroup as CompGroup

from c3po.control.control import Control as Control
from c3po.control.control import ControlSet as ControlSet

from c3po.control.generator import Device as Device
from c3po.control.generator import AWG as AWG
from c3po.control.generator import Mixer as Mixer
from c3po.control.generator import Generator as Generator

from c3po.optimizer.optimizer import Optimizer as Opt
from c3po.simulation.simulator import Simulator as Sim

###### Starting tensorflow session and setting log level
tf_log_level_info()
set_tf_log_level(3)
print("current log level: " + str(get_tf_log_level()))
# to make sure session is empty look up: tf.reset_default_graph()
sess = tf_setup()
print("Available tensorflow devices: ")
tf_list_avail_devices()
writer = tf.summary.FileWriter( './logs/optim_log', sess.graph)

###### Set up models: one assumed (initial), a real one and one to optimize
q1 = Qubit(
    name = "Q1",
    desc = "Qubit 1",
    comment = "The assumed qubit in this chip",
    freq = 6e9*2*np.pi,
    delta = 1e6 * 2 * np.pi,
    hilbert_dim = 2
    )
# r1 = Resonator(
#     name = "R1",
#     desc = "Resonator 1",
#     comment = "The resonator driving Qubit 1",
#     freq = 9e9*2*np.pi,
#     hilbert_dim = 5
#     )
# q1r1 = Coupling(
#     name = "Q1-R1",
#     desc = "Coupling between Resonator 1 and Qubit 1",
#     comment = " ",
#     connected = [q1.name, r1.name],
#     strength = 150e6*2*np.pi
#     )
drive = Drive(
    name = "D1",
    desc = "Drive 1",
    comment = "Drive line 1 on qubit 1",
    connected = [q1.name]
    )

q2 = Qubit(
    name = "Q2",
    desc = "Qubit 2",
    comment = "The actual qubit in this chip",
    freq = 6.05e9*2*np.pi,
    delta = 1e6 * 2 * np.pi,
    hilbert_dim = 2
    )

drive2 = Drive(
    name = "D2",
    desc = "Drive 2",
    comment = "Drive line 2 on qubit 2",
    connected = [q2.name]
    )

chip_elements = [
     q1,
     # r1,
     # q1r1,
     drive
     ]

chip2_elements = [
    q2,
    drive2
    ]

initial_model = mdl(chip_elements)
optimize_model = mdl(chip_elements)
real_model = mdl(chip2_elements)

###### Set up controls
env_group = CompGroup()
env_group.name = "env_group"
env_group.desc = "group containing all components of type envelop"
carr_group = CompGroup()
carr_group.name = "carr_group"
carr_group.desc = "group containing all components of type carrier"

carrier_parameters = {'freq' : 5.95e9 * 2 * np.pi}
carr = CtrlComp(
    name = "carrier",
    desc = "Frequency of the local oscillator",
    params = carrier_parameters,
    groups = [carr_group.get_uuid()]
)
carr_group.add_element(carr)

flattop_params1 = {
    'amp' : np.pi * 1.2 / 7e-9, # 448964342.3828554,
    'T_up' : 3e-9,
    'T_down' : 10e-9,
    'xy_angle' : 0.0,
    'freq_offset' : 0e6 * 2 * np.pi, #150782.0898206234,
}
params_bounds = {
    'amp' : [50e6 * 2 * np.pi, 100e6 * 2 * np.pi],
    'T_up' : [1e-9, 11e-9],
    'T_down' : [1e-9, 11e-9],
    'xy_angle' : [-np.pi, np.pi],
    'freq_offset' : [-0.2e9 * 2 * np.pi, 0.2e9 * 2 * np.pi]
}
def my_flattop(t, params):
    t_up = tf.cast(params['T_up'], tf.float64)
    t_down = tf.cast(params['T_down'], tf.float64)
    T2 = tf.maximum(t_up, t_down)
    T1 = tf.minimum(t_up, t_down)
    return (1 + tf.math.erf((t - T1) / 1e-9)) / 2 * \
            (1 + tf.math.erf((-t + T2) / 1e-9)) / 2
p1 = CtrlComp(
    name = "pulse1",
    desc = "flattop comp 1 of signal 1",
    shape = my_flattop,
    params = flattop_params1,
    bounds = params_bounds,
    groups = [env_group.get_uuid()]
)
env_group.add_element(p1)

comps = []
comps.append(carr)
comps.append(p1)
ctrl = Control()
ctrl.name = "control1"
ctrl.t_start = 0.0
ctrl.t_end = 12e-9
ctrl.comps = comps
ctrls = ControlSet([ctrl])

####### Set up signal generator
awg = AWG()
mixer = Mixer()
devices = {
    "awg" : awg,
    "mixer" : mixer
}
resolutions = {
    "awg" : 1e9,
    "sim" : 5e11
}
resources = [ctrl]
resource_groups = {
    "env" : env_group,
    "carr" : carr_group
}
gen = Generator()
gen.devices = devices
gen.resolutions = resolutions
gen.resources = resources
gen.resource_groups = resource_groups

###### Set up simulation objects
sim = Sim(initial_model, gen, ctrls)
exp_sim = Sim(real_model, gen, ctrls)
opt_sim = Sim(real_model, gen, ctrls)

##### Define fidelity functions
psi_init = np.array(
    [[1.+0.j],
     [0.+0.j]],
    )
psi_goal = np.array(
    [[0.+0.j],
     [1.+0.j]],
    )

# Goal to drive on qubit 1
# U_goal = np.array(
#     [[0.+0.j, 1.+0.j, 0.+0.j],
#      [1.+0.j, 0.+0.j, 0.+0.j],
#      [0.+0.j, 0.+0.j, 1.+0.j]]
#     )

def evaluate_signals_psi(pulse_params, opt_params):
    model_params = sim.model.params
    U = sim.propagation(pulse_params, opt_params, model_params)
    psi_actual = tf.matmul(U, psi_init)
    overlap = tf.matmul(psi_goal.T, psi_actual)
    return 1-tf.cast(tf.conj(overlap)*overlap, tf.float64)

def experiment_evaluate_psi(pulse_params, opt_params):
    model_params = exp_sim.model.params
    U = exp_sim.propagation(pulse_params, opt_params, model_params)
    psi_actual = tf.matmul(U, psi_init)
    overlap = tf.matmul(psi_goal.T, psi_actual)
    return 1-tf.cast(tf.conj(overlap)*overlap, tf.float64)

def match_model_psi(model_params, opt_params, measurements):
    model_error = 0
    measurements = measurements[-30::5]
    for m in measurements:
        pulse_params = m[0]
        result = m[1]
        U = sim.propagation(pulse_params, opt_params, model_params)
        psi_actual = tf.matmul(U, psi_init)
        overlap = tf.matmul(psi_goal.T, psi_actual)
        diff = (1-tf.cast(tf.conj(overlap)*overlap, tf.float64)) - result
        model_error += diff * diff
    return model_error

##### Define optimizer object
rechenknecht = Opt()
rechenknecht.store_history = True
rechenknecht.set_session(sess)
rechenknecht.set_log_writer(writer)

opt_map = {
    'amp' : [(ctrl.get_uuid(), p1.get_uuid())],
    # 'T_up' : [
    #     (ctrl.get_uuid(), p1.get_uuid())
    #     ],
    # 'T_down' : [
    #     (ctrl.get_uuid(), p1.get_uuid())
    #     ],
    # 'xy_angle' : [(ctrl.get_uuid(), p1.get_uuid())],
    'freq_offset' : [(ctrl.get_uuid(), p1.get_uuid())]
}

print(
"""
#######################
# Optimizing pulse... #
#######################
"""
)

def callback(xk):print(xk)
settings = {} #'maxiter': 5}
rechenknecht.optimize_controls(
    controls = ctrls,
    opt_map = opt_map,
    opt = 'lbfgs',
    #opt = 'tf_grad_desc',
    settings = settings,
    calib_name = 'openloop',
    eval_func = evaluate_signals_psi,
    callback = callback
    )

system('clear')
print(rechenknecht.results)

print(
"""
#######################
# Calibrating...      #
#######################
"""
)

initial_spread = [5e6*2*np.pi, 20e6*2*np.pi]
opt_settings = {
    'CMA_stds': initial_spread,
    'maxiter' : 20,
    'ftarget' : 1e-4,
    'popsize' : 10
}
rechenknecht.optimize_controls(
    controls = ctrls,
    opt_map = opt_map,
    opt = 'cmaes',
#    opt = 'tf_grad_desc',
    settings = opt_settings,
    calib_name = 'closedloop',
    eval_func = experiment_evaluate_psi
    )

system('clear')
print(rechenknecht.results)

print(
"""
#######################
# Matching model...   #
#######################
"""
)

settings = {'maxiter': 100}
rechenknecht.learn_model(
    optimize_model,
    eval_func = match_model_psi,
    settings = settings,
    optim_name = 'model_learn',
    meas_results = 'closedloop'
    )

system('clear')
print(rechenknecht.results)
