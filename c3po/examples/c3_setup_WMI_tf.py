"""C3PO configuration file"""

from numpy import pi

import numpy as np
import qutip as qt
import matplotlib.pyplot as plt

from qutip import *
import c3po
from c3po.utils.tf_utils import *
from c3po.evolution.propagation import *
# from c3po.utils.aux_functions import *


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



def plot_dynamics(u_list, ts, states):
    pop = []
    for si in states:
        for ti in range(len(ts)):
            pop.append(abs(u_list[ti][si][0] ** 2))
#        plt.plot(ts, pop)
    return pop


def plot_dynamics_sesolve(u_list, ts):
    pop = []
    for ti in range(len(ts)):
        pop.append(abs(u_list.states[ti].full().T[0] ** 2))
#    plt.plot(ts, pop)
    return pop



tf_log_level_info()

set_tf_log_level(2)

print("current log level: " + str(get_tf_log_level()))

sess = tf_setup()




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

initial_model = c3po.Model(
        initial_parameters,
        initial_couplings,
        initial_hilbert_space,
        "True"
        )

initial_model.set_tf_session(sess)


q1_X_gate = c3po.Gate('qubit_1', qt.sigmax(), c3po.utils.envelopes.flattop)

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



control_func = q1_X_gate.get_control_fields('initial')


H = initial_model.get_Hamiltonian(control_func)

# print(H)


ts = np.linspace(0, 50e-9, int(1e4))



U0 = tensor(
    qeye(2),
    qeye(5)
)

# print(U0)

u = propagate(initial_model, q1_X_gate, U0, ts, "pwc")




psi = []
for i in range(0,2):
    for j in range(0,5):
        psi.append(tensor(basis(2,i), basis(5,j)))

U = []
for i in range(0, 10):
    U_res = propagate(initial_model, q1_X_gate, psi[i], ts, 'qutip_sesolve')
    U.append(U_res)





pop1 = plot_dynamics(u, ts, [0])

pop2 = plot_dynamics_sesolve(U[0], ts)


fig = plt.figure(1)
sp1 = plt.subplot(211)
sp1.title.set_text("pwc 1e4")
plt.plot(ts, pop1)

sp2 = plt.subplot(212)
sp2.title.set_text("sesolve")
plt.plot(ts, pop2)

plt.show()

# """ Plotting control functions """

# plt.rcParams['figure.dpi'] = 100

# fu = list(map(control_func[0], ts))
# env = list(map(lambda t: q1_X_gate.envelope(t, 5e-9, 45e-9), ts))
# fig, axs = plt.subplots(2, 1)

# axs[0].plot(ts/1e-9, env)

# axs[1].plot(ts/1e-9, fu)
# plt.show()

"""
BSB_X_gate = Gate((q, r),
        qt.tensor(qt.sigmap(), qt.sigmap())
            + qt.tensor(qt.sigmam(), qt.sigmam())
        )
"""
