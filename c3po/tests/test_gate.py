"""C3PO test file"""

import qutip as qt
import numpy as np
from numpy import pi as pi
import tensorflow as tf
from c3po.main.gate import Gate as gt
from c3po.utils.envelopes import flattop
from c3po.utils.tf_utils import tf_setup
"""
Device specific setup goes here. We need to provide a function that takes a
gate, evaluates it in the physical machine and returns a figure of merit.
"""


X_gate = gt('qubit_1', qt.sigmax())


def my_flattop(t, idx, guess):
    t_up = guess[idx['t_up']]
    t_down = guess[idx['t_down']]
    T2 = tf.maximum(t_up, t_down)
    T1 = tf.minimum(t_up, t_down)
    return (1 + tf.erf((t - T1) / 1e-9)) / 2 * \
           (1 + tf.erf((-t + T2) / 1e-9)) / 2


handmade_pulse = {
        'control1': {
            'carrier1': {
                'freq': 6e9*2*pi,
                'pulses': {
                    'pulse': {
                        'params': {
                            'amp': 15e6*2*pi,
                            't_up': 5e-9,
                            't_down': 45e-9,
                            'xy_angle': 0,
                            'freq_offset': 0e6*2*pi
                            },
                        'func': my_flattop
                        },
                    'drag': {
                        'params': {
                            'amp': 3e6*2*pi,
                            't_up': 25e-9,
                            't_down': 30e-9,
                            'xy_angle': pi/2,
                            'freq_offset': 0e6*2*pi
                            },
                        'func': my_flattop
                        }
                    }
                }
            }
        }

pulse_bounds = {
        'control1': {
            'carrier1': {
                'pulses': {
                    'pulse': {
                        'params': {
                            't_up': [2e-9, 98e-9],
                            't_down': [2e-9, 98e-9],
                            'freq_offset': [-1e9*2*pi, 1e9*2*pi]
                            }
                        }
                    }
                }
            }
        }

X_gate.set_parameters('initial', handmade_pulse)
X_gate.set_bounds(pulse_bounds)



print(X_gate.idxes)
print(X_gate.opt_idxes)"""C3PO test file"""

import qutip as qt
import numpy as np
from numpy import pi as pi
import tensorflow as tf
from c3po.main.gate import Gate as gt
from c3po.utils.envelopes import flattop
from c3po.utils.tf_utils import tf_setup
"""
Device specific setup goes here. We need to provide a function that takes a
gate, evaluates it in the physical machine and returns a figure of merit.
"""


X_gate = gt('qubit_1', qt.sigmax())


def my_flattop(t, idx, guess):
    t_up = guess[idx['t_up']]
    t_down = guess[idx['t_down']]
    T2 = tf.maximum(t_up, t_down)
    T1 = tf.minimum(t_up, t_down)
    return (1 + tf.erf((t - T1) / 1e-9)) / 2 * \
           (1 + tf.erf((-t + T2) / 1e-9)) / 2


handmade_pulse = {
        'control1': {
            'carrier1': {
                'freq': 6e9*2*pi,
                'pulses': {
                    'pulse': {
                        'params': {
                            'amp': 15e6*2*pi,
                            't_up': 5e-9,
                            't_down': 45e-9,
                            'xy_angle': 0,
                            'freq_offset': 0e6*2*pi
                            },
                        'func': my_flattop
                        },
                    'drag': {
                        'params': {
                            'amp': 3e6*2*pi,
                            't_up': 25e-9,
                            't_down': 30e-9,
                            'xy_angle': pi/2,
                            'freq_offset': 0e6*2*pi
                            },
                        'func': my_flattop
                        }
                    }
                }
            }
        }

pulse_bounds = {
        'control1': {
            'carrier1': {
                'pulses': {
                    'pulse': {
                        'params': {
                            't_up': [2e-9, 98e-9],
                            't_down': [2e-9, 98e-9],
                            'freq_offset': [-1e9*2*pi, 1e9*2*pi]
                            }
                        }
                    }
                }
            }
        }

X_gate.set_parameters('initial', handmade_pulse)
X_gate.set_bounds(pulse_bounds)



print(X_gate.idxes)
print(X_gate.opt_idxes)

tf_sess = tf_setup()

print(tf_sess.run(X_gate.to_scale_one('initial')))

print(X_gate.get_IQ('initial'))
print(tf_sess.run(X_gate.get_control_fields('initial')))


tf_sess = tf_setup()

print(tf_sess.run(X_gate.to_scale_one('initial')))

print(X_gate.get_IQ('initial'))
print(tf_sess.run(X_gate.get_control_fields('initial')))
