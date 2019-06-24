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


X_gate = gt('qubit_1', qt.sigmax(), tf_sess)


handmade_pulse = {
        'control1': {
            'carrier1': {
                'freq': 6e9*2*pi,
                # 'target': 'q1', # add here?
                'pulses': {
                    'pulse1': {
                        'amp': 15e6*2*pi,
                        't_up': 5e-9,
                        't_down': 45e-9,
                        'xy_angle': 0
                        }
                    }
                }
            }
        }

pulse_bounds = {
        'control1': {
            'carrier1': {
                'freq': [1e9*2*pi, 15e9*2*pi],
                'pulses': {
                    'pulse1': {
                        'amp':  [1e3*2*pi, 10e9*2*pi],
                        't_up': [2e-9, 98e-9],
                        't_down': [2e-9, 98e-9],
                        'xy_angle': [-pi, pi]
                        }
                    }
                }
            }
        }

X_gate.set_parameters('initial', handmade_pulse)
X_gate.set_bounds(pulse_bounds)

fields = X_gate.get_control_fields('initial')

ts = tf.cast(tf.linspace(0e-9, 50e-9, 1000), tf.float64)

c = fields[0](ts)

grads = jacobian(c, X_gate.parameters['initial'])

f = tf_sess.run(c)
g = tf_sess.run(grads)

plt.rcParams['figure.dpi'] = 100
p = tf_sess.run(X_gate.parameters['initial'])
n_params = p.shape[0]
fig, axs = plt.subplots(n_params+1, 1)
axs[0].plot(tf_sess.run(ts)/1e-9, f)
axs[0].set_ylabel('Signal')
labels = ['freq']
[labels.append(s) for s in X_gate.props]
for ii in range(n_params):
    axs[ii+1].plot(tf_sess.run(ts)/1e-9, g[:, ii])
    axs[ii+1].set_ylabel(labels[ii])

plt.show(block=False)

tf_sess.close()
