from c3po.control.envelopes import *
from c3po.cobj.component import ControlComponent as CtrlComp
from c3po.cobj.group import ComponentGroup as CompGroup
from c3po.control.control import Control as Control
from c3po.control.control import ControlSet as ControlSet

from c3po.control.generator import Device as Device
from c3po.control.generator import AWG as AWG
from c3po.control.generator import Mixer as Mixer
from c3po.control.generator import Generator as Generator

from c3po.utils.tf_utils import *

from c3po.cobj.component import *
from c3po.main.model import Model as mdl

from c3po.optimizer.optimizer import Optimizer as Opt
from c3po.simulation.simulator import Simulator as Sim

import uuid
import copy
import pickle

import tensorflow as tf

import matplotlib.pyplot as plt


#########################
# USER FRONTEND SECTION #
#########################

redo_closed_loop = True
redo_open_loop = True

qubit_freq = 5e9*2*np.pi
qubit_anhar = -300e6 *2*np.pi
#qubit_freq = 31730085796.517048 # from previous optim
qubit_lvls = 3


drive_amp = 40e-3 # 100 mV

mV_to_Amp = 2e9*np.pi

#mV_to_Amp = 4523893481.479533 # from previous optim
qubit_g = np.zeros([qubit_lvls, 1])
qubit_g[0] = 1

qubit_e = np.zeros([qubit_lvls, 1])
qubit_e[1] = 1

psi_init = tf.constant(qubit_g, tf.complex128)
psi_goal = tf.constant(qubit_e.T, tf.complex128)

##########################
#    END USER SECTION    #
##########################

env_group = CompGroup()
env_group.name = "env_group"
env_group.desc = "group containing all components of type envelop"


carr_group = CompGroup()
carr_group.name = "carr_group"
carr_group.desc = "group containing all components of type carrier"


carrier_parameters = {
    'freq' : 5e9 * 2 * np.pi
}

carr = CtrlComp(
    name = "carrier",
    desc = "Frequency of the local oscillator",
    params = carrier_parameters,
    groups = [carr_group.get_uuid()]
)
carr_group.add_element(carr)


flattop_params1 = {
    'amp' : drive_amp,
    'T_up' : 0.5e-9,
    'T_down' : 8e-9,
    'xy_angle' : 0.0,
    'detuning' : -300e6 * 2 * np.pi,
    'freq_offset' : 0e6 * 2 * np.pi
}

params_bounds = {
    'amp' : [10e-3, 250e-3],
    'T_up' : [1e-9, 22e-9],
    'T_down' : [1e-9, 22e-9],
    'xy_angle' : [-np.pi, np.pi],
    'detuning' :  [-500e6 * 2 * np.pi, -100e6 * 2 * np.pi],
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
ctrl.t_end = 9e-9
ctrl.comps = comps


ctrls = ControlSet([ctrl])

opt_map = {
    'amp' : [(ctrl.get_uuid(), p1.get_uuid())],
#    'T_up' : [(ctrl.get_uuid(), p2.get_uuid())],
#    'T_down' : [(ctrl.get_uuid(), p2.get_uuid())],
    'xy_angle' : [(ctrl.get_uuid(), p1.get_uuid())],
    'freq_offset' : [(ctrl.get_uuid(), p1.get_uuid())],
    'detuning' : [(ctrl.get_uuid(), p1.get_uuid())]
}

awg = AWG()
awg.options = "drag"

mixer = Mixer()


devices = {
    "awg" : awg,
    "mixer" : mixer
}

resolutions = {
    "awg" : 1e9,
    "sim" : 5e10
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

output = gen.generate_signals()


set_tf_log_level(3)


q1 = Qubit(
    name = "Q1",
    desc = "Qubit 1",
    comment = "The one and only qubit in this chip",
    freq = qubit_freq,
    delta = qubit_anhar,
    hilbert_dim = qubit_lvls
    )

drive = Drive(
    name = "D1",
    desc = "Drive 1",
    comment = "Drive line 1 on qubit 1",
    connected = [q1.name]
    )

chip_elements = [
    q1,
     drive
     ]

initial_model = mdl(chip_elements, mV_to_Amp)
optimize_model = mdl(chip_elements, mV_to_Amp)

q2 = Qubit(
    name = "Q1",
    desc = "Qubit 2",
    comment = "The one and only qubit in this chip",
    freq = 5.05e9*2*np.pi,
    delta = 1.1*qubit_anhar,
    hilbert_dim = qubit_lvls
    )

drive2 = Drive(
    name = "D2",
    desc = "Drive 2",
    comment = "Drive line 1 on qubit 1",
    connected = [q2.name]
    )

chip2_elements = [
    q2,
    drive2
    ]

real_model = mdl(chip2_elements, 0.72*2e9*np.pi)

rechenknecht = Opt()


print("Available tensorflow devices: ")
tf_list_avail_devices()



opt_params = ctrls.get_corresponding_control_parameters(opt_map)
rechenknecht.opt_params = opt_params

sim = Sim(initial_model, gen, ctrls)

# Goal to drive on qubit 1
# U_goal = np.array(
#     [[0.+0.j, 1.+0.j, 0.+0.j],
#      [1.+0.j, 0.+0.j, 0.+0.j],
#      [0.+0.j, 0.+0.j, 1.+0.j]]
#     )

sim.model = optimize_model

exp_sim = Sim(real_model, gen, ctrls)

rechenknecht.simulate_noise = True

def evaluate_signals(pulse_params, opt_params):

    model_params = sim.model.params
    U = sim.propagation(pulse_params, opt_params, model_params)
    psi_actual = tf.matmul(U, psi_init)
    overlap = tf.matmul(psi_goal, psi_actual)

    return 1-tf.cast(tf.linalg.adjoint(overlap)*overlap, tf.float64)

def experiment_evaluate(pulse_params, opt_params):
    model_params = exp_sim.model.params
    U = exp_sim.propagation(pulse_params, opt_params, model_params)
    psi_actual = tf.matmul(U, psi_init)
    overlap = tf.matmul(psi_goal, psi_actual)
    return 1-tf.cast(tf.linalg.adjoint(overlap)*overlap, tf.float64)

def match_model_psi(model_params, opt_params, pulse_params, result):

    U = sim.propagation(pulse_params, opt_params, model_params)

    psi_actual = tf.matmul(U, psi_init)
    overlap = tf.matmul(psi_goal, psi_actual)
    diff = (1-tf.cast(tf.linalg.adjoint(overlap)*overlap, tf.float64)) - result

    model_error = diff * diff

    return model_error

print(
"""
#######################
# Optimizing pulse... #
#######################
"""
)

def callback(xk):
    print(xk)

settings = {} #'maxiter': 1}

rechenknecht.optimize_controls(
    controls = ctrls,
    opt_map = opt_map,
    opt = 'lbfgs',
#    opt = 'tf_grad_desc',
    settings = settings,
    calib_name = 'open_loop',
    eval_func = evaluate_signals,
    callback = callback
    )

sim.plot_dynamics(psi_init)

initial_spread = [5e-3, 0.1, 20e6*2*np.pi, 20e6*2*np.pi]

opt_settings = {
    'CMA_stds': initial_spread,
    'maxiter' : 20,
#    'ftarget' : 1e-4,
    'popsize' : 5
}

if redo_closed_loop:
    print(
    """
    #######################
    # Calibrating...      #
    #######################
    """

    )

    rechenknecht.optimize_controls(
        controls = ctrls,
        opt_map = opt_map,
        opt = 'cmaes',
    #    opt = 'tf_grad_desc',
        settings = opt_settings,
        calib_name = 'closed_loop',
        eval_func = experiment_evaluate
        )

# opt_sim = Sim(real_model, gen, ctrls)
# Fed: this here really scared me for a second

settings = {'maxiter': 100}

print(
"""
    #######################
    # Matching model...   #
    #######################
"""
)

if not redo_closed_loop:
    rechenknecht.load_history(previous_optim_log)

rechenknecht.learn_model(
    optimize_model,
    eval_func = match_model_psi,
    settings = settings,
    )
