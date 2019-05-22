"""C3PO configuration file"""

from numpy import pi

import numpy as np
import qutip as qt
import matplotlib.pyplot as plt

from qutip import *
import c3po
from c3po.main.model import Model as mdl
from c3po.main.gate import Gate as gt



"""
This is  disabled for now. The idea is to generalize the setup part later and
use the System class to construct a model.


q = components.qubit
q.set_name('qubit_1')
r = components.resonator
r.set_name('cavity')
q_drv = components.control
q_drv.set_name('qubit_drive')

couplings = [
        (q, r)
        ]
controls = [
        (q, q_drv)
        ]

WMI_memory = System([q, r, q_drv], couplings, controls)

"""


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

model_init = [
        initial_parameters,
        initial_couplings,
        initial_hilbert_space
        ]

initial_model = mdl(
        initial_parameters,
        initial_couplings,
        initial_hilbert_space,
        )



handmade_pulse = {
        'control1': {
            'carrier1': {
                'freq': 6e9*2*pi,
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


q1_X_gate = gt('qubit_1', qt.sigmax())
q1_X_gate.set_parameters('initial', handmade_pulse)

crazy_pulse = {
        'control1': {
            'carrier1': {
                'freq': 6e9*2*pi,
                'pulses': {
                    'pulse1': {
                        'amp': 15e6*2*pi,
                        't_up': 5e-9,
                        't_down': 45e-9,
                        'xy_angle': 0
                        }
                    }
                },
            'carrier2': {
                'freq': 6e9*2*pi,
                'pulses': {
                    'pulse1': {
                        'amp': 15e6*2*pi,
                        't_up': 5e-9,
                        't_down': 45e-9,
                        'xy_angle': 0
                    },
                    'pulse2': {
                        'amp': 20e6*2*pi,
                        't_up': 10e-9,
                        't_down': 4e-9,
                        'xy_angle': pi/2
                        }
                    }
                }
            }
        }



control_func = [q1_X_gate.get_control_fields('initial')]


H = initial_model.get_Hamiltonian(control_func)

print(H)


""" plotting control functions """

ts = np.linspace(0, 50e-9, int(1e4))

fu = list(map(control_func[0], ts))
env = list(map(lambda t: q1_X_gate.envelope(t, 5e-9, 45e-9), ts))
fig, axs = plt.subplots(2, 1)

axs[0].plot(ts/1e-9, env)

axs[1].plot(ts/1e-9, fu)
plt.show()

"""
BSB_X_gate = Gate((q, r),
        qt.tensor(qt.sigmap(), qt.sigmap())
            + qt.tensor(qt.sigmam(), qt.sigmam())
        )
"""
