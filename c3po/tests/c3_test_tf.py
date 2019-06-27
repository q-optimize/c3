"""C3PO configuration file"""

from numpy import pi
import qutip as qt
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.python.ops.parallel_for.gradients import jacobian
from c3po.main.gate import Gate as gt
from c3po.utils.tf_utils import tf_setup

"""
Device specific setup goes here. We need to provide a function that takes a
gate, evaluates it in the physical machine and returns a figure of merit.
"""

tf_sess = tf_setup()


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


x0 = tf.constant(X_gate.parameters['initial'])
fields, ts = X_gate.get_control_fields(x0, 10e9)

c = fields[0]

grads = jacobian(c, x0)

X_gate.idxes

f = tf_sess.run(c)
g = tf_sess.run(grads)

plt.rcParams['figure.dpi'] = 100
p = tf_sess.run(X_gate.parameters['initial'])
n_params = p.shape[0]
fig, axs = plt.subplots(n_params+1, 1)
axs[0].plot(tf_sess.run(ts)/1e-9, f)
axs[0].set_ylabel('Signal')
labels = ['amp', 'freq_offset', 't_down', 't_up', 'xy_angle']
for ii in range(n_params):
    axs[ii+1].plot(tf_sess.run(ts)/1e-9, g[:, ii])
    axs[ii+1].set_ylabel(labels[ii])

plt.show(block=False)

tf_sess.close()
