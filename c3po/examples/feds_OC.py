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


# System

qubit_freq = 6e9 * 2 * np.pi
qubit_anhar = -100e6 *2 * np.pi
qubit_lvls = 6

mV_to_Amp = 2e9*np.pi

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

simple_model = mdl(chip_elements, mV_to_Amp)

# Control

def drag(t, params):
    T = params['T']
    sigma = tf.cast(T / 6, tf.float64)
    norm = (tf.sqrt(2 * np.pi * sigma ** 2) * tf.math.erf(T / (np.sqrt(8) * sigma)) \
     - T * tf.exp(-T ** 2 / (8 * sigma ** 2)))
    B = tf.exp(-T ** 2 / (8 * sigma ** 2))
    return (tf.exp(-(t - T / 2) ** 2 / (2 * sigma ** 2)) - B) ** 2 / norm

def drag_der(t, params):
    T = params['T']
    sigma = tf.cast(T / 6, tf.float64)
    norm = (tf.sqrt(2 * np.pi * sigma ** 2) * tf.math.erf(T / (np.sqrt(8) * sigma)) \
     - T * tf.exp(-T ** 2 / (8 * sigma ** 2)))
    B = tf.exp(-T ** 2 / (8 * sigma ** 2))
    return - 2 * (tf.exp(-(t - T / 2) ** 2 / (2 * sigma ** 2)) - B)* \
            (np.exp(-(t - T / 2) ** 2 / (2 * sigma ** 2))) * (t - T / 2) / sigma ** 2 / norm

def gaussian(t, params):
    T = params['T']
    sigma = tf.cast(T / 6, tf.float64)
    norm = (tf.sqrt(2 * np.pi * sigma ** 2) * tf.math.erf(T / (np.sqrt(8) * sigma)) \
     - T * tf.exp(-T ** 2 / (8 * sigma ** 2)))
    B = tf.exp(-T ** 2 / (8 * sigma ** 2))
    return (tf.exp(-(t - T / 2) ** 2 / (2 * sigma ** 2)) - B) / norm

def gaussian_der(t, params):
    T = params['T']
    sigma = tf.cast(T / 6, tf.float64)
    norm = (tf.sqrt(2 * np.pi * sigma ** 2) * tf.math.erf(T / (np.sqrt(8) * sigma)) \
     - T * tf.exp(-T ** 2 / (8 * sigma ** 2)))
    return - (np.exp(-(t - T / 2) ** 2 / (2 * sigma ** 2))) * (t - T / 2) / sigma ** 2 / norm


pulse_params = {
        'amp' : np.pi / mV_to_Amp,
        'T' : 8e-9,
        'xy_angle' : 0.0,
        'freq_offset' : 0e6 * 2 * np.pi
    }

corr_params = {
        'amp' : np.pi / mV_to_Amp/qubit_anhar,
        'T' : 8e-9,
        'xy_angle' : np.pi/2,
        'freq_offset' : 0e6 * 2 * np.pi
    }

params_bounds = {
        'amp' : [0.01* np.pi / mV_to_Amp, 1.5 * np.pi / mV_to_Amp] ,
        'T' : [7e-9, 12e-9],
        'xy_angle' : [-1 * np.pi/2,1  * np.pi/2],
        'freq_offset' : [-100* 1e6 * 2 * np.pi, 100* 1e6 * 2 * np.pi]
    }

corr_bounds = {
        'amp' : [0.001* np.pi / mV_to_Amp/qubit_anhar,
                 3 * np.pi / mV_to_Amp/qubit_anhar] ,
        'T' : [7e-9, 12e-9],
        'xy_angle' : [-1 * np.pi/2,1  * np.pi/2],
        'freq_offset' : [-100* 1e6 * 2 * np.pi, 100* 1e6 * 2 * np.pi]
    }

carrier_parameters = {
    'freq' : 5.95e9 * 2 * np.pi
}


env_group = CompGroup()
env_group.name = "env_group"
env_group.desc = "group containing all components of type envelop"
carr_group = CompGroup()
carr_group.name = "carr_group"
carr_group.desc = "group containing all components of type carrier"

p1 = CtrlComp(
    name = "pulse1",
    desc = "Gaussian comp 1 of signal 1",
    shape = drag,
    params = pulse_params,
    bounds = params_bounds,
    groups = [env_group.get_uuid()]
)

p2 = CtrlComp(
    name = "pulse2",
    desc = "Gaussian comp 2 of signal 1",
    shape = drag_der,
    params = corr_params,
    bounds = corr_bounds,
    groups = [env_group.get_uuid()]
)

env_group.add_element(p1)
env_group.add_element(p2)

carr = CtrlComp(
    name = "carrier",
    desc = "Frequency of the local oscillator",
    params = carrier_parameters,
    groups = [carr_group.get_uuid()]
)
carr_group.add_element(carr)

comps = []
comps.append(carr)
comps.append(p1)
comps.append(p2)

ctrl = Control()
ctrl.name = "control1"
ctrl.t_start = 0.0
ctrl.t_end = 12e-9
ctrl.comps = comps

ctrls = ControlSet([ctrl])


awg = AWG()
mixer = Mixer()


devices = {
    "awg" : awg,
    "mixer" : mixer
}

resolutions = {
    "awg" : 1.2e9,
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

sim = Sim(simple_model, gen, ctrls)

plt.rcParams['figure.dpi'] = 100
fig, axs = plt.subplots(1, 1)
plt.ion()
sim.fig = fig
sim.axs = axs

opt = Opt()

qubit_g = np.zeros([qubit_lvls, 1])
qubit_g[0] = 1

qubit_e = np.zeros([qubit_lvls, 1])
qubit_e[1] = 1

ket_init = tf.constant(qubit_g, tf.complex128)
bra_goal = tf.constant(qubit_e.T, tf.complex128)

def evaluate_signals(pulse_params, opt_params):

    model_params = sim.model.params
    U = sim.propagation(pulse_params, opt_params, model_params)
    psi_actual = tf.matmul(U, ket_init)
    overlap = tf.matmul(bra_goal, psi_actual)

    return 1-tf.cast(tf.math.conj(overlap)*overlap, tf.float64)

opt_map = {
    'amp' : [(ctrl.get_uuid(), p1.get_uuid()),
             (ctrl.get_uuid(), p2.get_uuid())],
    'freq_offset' : [(ctrl.get_uuid(), p1.get_uuid()),
                     (ctrl.get_uuid(), p2.get_uuid())],
    'T' : [(ctrl.get_uuid(), p1.get_uuid()),
                     (ctrl.get_uuid(), p2.get_uuid())],
    'xy_angle' : [(ctrl.get_uuid(), p1.get_uuid()),
                     (ctrl.get_uuid(), p2.get_uuid())]
}

opt_params = ctrls.get_corresponding_control_parameters(opt_map)

opt.optimize_controls(
    controls = ctrls,
    opt_map = opt_map,
    opt = 'lbfgs',
    calib_name = 'openloop',
    eval_func = evaluate_signals
    )
