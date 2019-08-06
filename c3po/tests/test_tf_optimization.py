"""C3PO configuration file"""

from numpy import pi

import numpy as np
import qutip as qt
import matplotlib.pyplot as plt

from qutip import *
import c3po

from c3po.main.model import Model as mdl
from c3po.main.gate import Gate as gt
from c3po.fidelity.measurement import Simulation as sim

from c3po.utils.tf_utils import *
from c3po.evolution.propagation import *

import time

from tensorflow.python import debug as tf_debug

tf_log_level_info()

set_tf_log_level(2)

print("current log level: " + str(get_tf_log_level()))

sess = tf_setup()

print(" ")
print("Available tensorflow devices: ")
tf_list_avail_devices()



initial_parameters = {
        'qubit_1': {'freq': 6e9*2*pi},
        'cavity': {'freq': 9e9*2*pi}
        }

initial_couplings = {
        'q1_cav': {'strength': 150e6*2*pi}
        }

initial_hilbert_space = {
        'qubit_1': 2,
        'cavity': 5
        }

initial_model = mdl(
        initial_parameters,
        initial_couplings,
        initial_hilbert_space,
        "True"
        )

initial_model.set_tf_session(sess)


def my_flattop(t, idx, guess):
    t_up = guess[idx['t_up']]
    t_down = guess[idx['t_down']]
    T2 = tf.maximum(t_up, t_down)
    T1 = tf.minimum(t_up, t_down)
    return (1 + tf.erf((t - T1) / 2e-9)) / 2 * \
           (1 + tf.erf((-t + T2) / 2e-9)) / 2

handmade_pulse = {
        'control1': {
            'carrier1': {
                'freq': 6e9*2*pi,
                'pulses': {
                    'pulse': {
                        'params': {
                            'amp': 40e6*2*pi,
                            't_up': 5e-9,
                            't_down': 25e-9,
                            'xy_angle': 0,
                            'freq_offset': 0e6*2*pi
                            },
                        'func': my_flattop
                        }
                    }
                }
            }
        }


U_goal = tensor(
    basis(2,1),
    basis(5,0)
).full()

U0 = tensor(
    basis(2,0),
    basis(5,0)
).full()

X_gate = gt('qubit_1', U_goal, T_final=50e-9)
pulse_bounds = {
        'control1': {
            'carrier1': {
                'pulses': {
                    'pulse': {
                        'params': {
                            'amp': [15e6*2*pi, 65e6*2*pi],
                            't_up': [2e-9, 48e-9],
                            't_down': [2e-9, 48e-9],
                            'xy_angle': [-pi, pi],
                            'freq_offset': [-20e6*2*pi, 20e6*2*pi]
                            }
                        }
                    }
                }
            }
        }

X_gate.set_parameters('initial', handmade_pulse)
X_gate.set_bounds(pulse_bounds)

rechenknecht = sim(initial_model, sesolve_pwc, sess)

res = 50e9
rechenknecht.resolution=res

#sess = tf_debug.LocalCLIDebugWrapperSession(sess) # Enable this to debug
rechenknecht.optimize_gate(U0, X_gate, sess)
