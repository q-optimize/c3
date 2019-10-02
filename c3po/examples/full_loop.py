import time
import uuid
import copy
import pickle
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

###### Repeat full loop or load data and only do model learning
save_optim_log = 'learned_freq_inhar_resp4.pickle'
previous_optim_log = 'learn_freq_inhar_resp4.pickle'
redo_open_loop = True
redo_closed_loop = True

###### User Input
qubit_freq = 6e9*2*np.pi
qubit_inharm = -300e6 * 2 * np.pi
qubit_lvls = 3
drive_amp = 88e-3 # 80 mV
mV_to_Amp = 1e9*2*np.pi

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
q1_sim = Qubit(
    name = "Q1sim",
    desc = "Qubit 1 in simulation",
    comment = "The assumed qubit in this chip",
    freq = qubit_freq * 1.01,
    delta = qubit_inharm * 0.95,
    hilbert_dim = qubit_lvls
    )
drive_sim = Drive(
    name = "D1sim",
    desc = "Drive 1",
    comment = "Drive line 1 on qubit 1",
    connected = [q1_sim.name]
    )

q1_exp = Qubit(
    name = "Q1exp",
    desc = "Qubit 1 in experiment",
    comment = "The actual qubit in this chip",
    freq = qubit_freq,
    delta = qubit_inharm,
    hilbert_dim = qubit_lvls
    )
drive_exp = Drive(
    name = "D1exp",
    desc = "Drive 1",
    comment = "Drive line 1 on qubit 1",
    connected = [q1_exp.name]
    )

chip_elements_sim = [
     q1_sim,
     drive_sim
     ]

chip_elements_exp = [
    q1_exp,
    drive_exp
    ]

initial_model = mdl(chip_elements_sim, mV_to_Amp * 0.9)
optimize_model = mdl(chip_elements_sim, mV_to_Amp * 0.9)
real_model = mdl(chip_elements_exp, mV_to_Amp)

###### Set up controls
env_group = CompGroup()
env_group.name = "env_group"
env_group.desc = "group containing all components of type envelop"
carr_group = CompGroup()
carr_group.name = "carr_group"
carr_group.desc = "group containing all components of type carrier"

carrier_parameters = {'freq' : qubit_freq}
carr = CtrlComp(
    name = "carrier",
    desc = "Frequency of the local oscillator",
    params = carrier_parameters,
    groups = [carr_group.get_uuid()]
)
carr_group.add_element(carr)

flattop_params = {
    'amp' : drive_amp,
    'T_up' : 3e-9,
    'T_down' : 9e-9,
    'xy_angle' : 0.0,
    'freq_offset' : 0e6 * 2 * np.pi
}
drag_left = {
    'amp' : drive_amp * 0.1,
    'T_up' : 2e-9,
    'T_down' : 4e-9,
    'xy_angle' : np.pi/2,
    'freq_offset' : 0e6 * 2 * np.pi
}
drag_right = {
    'amp' : drive_amp * 0.1,
    'T_up' : 8e-9,
    'T_down' : 10e-9,
    'xy_angle' : -np.pi/2,
    'freq_offset' : 0e6 * 2 * np.pi
}
params_bounds = {
    'amp' : [1e-3, 150e-3],
    'T_up' : [3e-9, 13e-9],
    'T_down' : [3e-9, 13e-9],
    'xy_angle' : [-np.pi, np.pi],
    'freq_offset' : [-0.1e9 * 2 * np.pi, 0.1e9 * 2 * np.pi]
}
def my_flattop(t, params):
    t_up = tf.cast(params['T_up'], tf.float64)
    t_down = tf.cast(params['T_down'], tf.float64)
    T2 = tf.maximum(t_up, t_down)
    T1 = tf.minimum(t_up, t_down)
    return (1 + tf.math.erf((t - T1) / 1e-9)) / 2 * \
            (1 + tf.math.erf((-t + T2) / 1e-9)) / 2
p1 = CtrlComp(
    name = "Pulse 1",
    desc = "flattop comp 1 of signal 1",
    shape = my_flattop,
    params = flattop_params,
    bounds = params_bounds,
    groups = [env_group.get_uuid()]
)
dragL = CtrlComp(
    name = "Drag Left",
    desc = "left part of the drag correction of the flat top",
    shape = my_flattop,
    params = drag_left,
    bounds = params_bounds,
    groups = [env_group.get_uuid()]
)
dragR = CtrlComp(
    name = "Drag Right",
    desc = "right part of the drag correction of the flat top",
    shape = my_flattop,
    params = drag_right,
    bounds = params_bounds,
    groups = [env_group.get_uuid()]
)
env_group.add_element(p1)
env_group.add_element(dragL)
env_group.add_element(dragR)

comps = []
comps.append(carr)
comps.append(p1)
comps.append(dragL)
comps.append(dragR)
ctrl = Control()
ctrl.name = "control1"
ctrl.t_start = 0.0
ctrl.t_end = 16e-9
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
opt_sim = Sim(optimize_model, gen, ctrls)

##### Define fidelity functions
psi_init = np.array(
    [[1.+0.j],
     [0.+0.j],
     [0.+0.j]],
    )
psi_goal = np.array(
    [[0.+0.j],
     [1.+0.j],
     [0.+0.j]],
    )

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

def match_model_psi(model_params, opt_params, pulse_params, result):
    U = opt_sim.propagation(pulse_params, opt_params, model_params)
    psi_actual = tf.matmul(U, psi_init)
    overlap = tf.matmul(psi_goal.T, psi_actual)
    diff = (1-tf.cast(tf.conj(overlap)*overlap, tf.float64)) - result
    model_error = diff * diff
    return model_error

##### Define optimizer object
rechenknecht = Opt()
rechenknecht.set_session(sess)
rechenknecht.set_log_writer(writer)

opt_map = {
    'amp' : [(ctrl.get_uuid(), p1.get_uuid()),
             (ctrl.get_uuid(), dragL.get_uuid()),
             (ctrl.get_uuid(), dragR.get_uuid())],
    #'T_up' : [(ctrl.get_uuid(), p1.get_uuid())],
    'T_down' : [ (ctrl.get_uuid(), p1.get_uuid()) ],
    'freq_offset' : [(ctrl.get_uuid(), p1.get_uuid())]
}
opt_params = ctrls.get_corresponding_control_parameters(opt_map)
rechenknecht.opt_params = opt_params
initial_spread = [10e-3, 2e-3, 2e-3, 1e-9, 5e6*2*np.pi]

if redo_open_loop:
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
        settings = settings,
        calib_name = 'open_loop',
        eval_func = evaluate_signals_psi,
        callback = callback
        )
    # system('clear')
    # print(rechenknecht.results)
    rechenknecht.save_history(save_optim_log)

if redo_closed_loop:
    if not redo_open_loop:
        rechenknecht.load_history(previous_optim_log)
    rechenknecht.simulate_noise = True
    print(
    """
    #######################
    # Calibrating...      #
    #######################
    """
    )
    opt_settings = {
        'CMA_stds': initial_spread,
        'maxiter' : 50,
        'ftarget' : 1e-4,
        'popsize' : 20
    }
    rechenknecht.optimize_controls(
        controls = ctrls,
        opt_map = opt_map,
        opt = 'cmaes',
        settings = opt_settings,
        calib_name = 'closed_loop',
        eval_func = experiment_evaluate_psi
        )
    # system('clear')
    # print(rechenknecht.results)
    rechenknecht.save_history(save_optim_log)

print(
"""
#######################
# Matching model...   #
#######################
"""
)
if not redo_closed_loop:
    rechenknecht.load_history(previous_optim_log)
rechenknecht.random_samples = True
rechenknecht.shai_fid = False
settings = {'maxiter': 100}
rechenknecht.learn_model(
    model = optimize_model,
    eval_func = match_model_psi,
    settings = settings
    )
# system('clear')
# print(rechenknecht.results)
rechenknecht.save_history(save_optim_log)
