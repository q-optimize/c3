"""C3PO configuration file"""

from numpy import pi

import numpy as np
import qutip as qt
import matplotlib.pyplot as plt

from qutip import *
import c3po

from c3po.main.model import Model as mdl
from c3po.main.gate import Gate as gt

from c3po.utils.tf_utils import *
from c3po.evolution.propagation import *


#####
# hacked plotting functions

def plot_dynamics(u_list, ts, states):
    pop = []
    for si in states:
        for ti in range(len(ts)):
            pop.append(abs(u_list[ti][si][0] ** 2))
    return pop


def plot_dynamics_sesolve(u_list, ts):
    pop = []
    for ti in range(len(ts)):
        pop.append(abs(u_list.states[ti].full().T[0] ** 2))
    return pop

#####




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
        "False"
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



####### 
# actually not needed just to be able to show the hamilton before propagation

# control_func = [q1_X_gate.get_control_fields('initial')]
# H = initial_model.get_Hamiltonian(control_func)
# print(H)

######


ts = np.linspace(0, 50e-9, int(1e4))


#####
# pwc test

U0 = tensor(
    qeye(2),
    qeye(5)
)

u = propagate(initial_model, q1_X_gate, U0, ts, "pwc")

#####


#####
# sesolve for reference

psi = []
for i in range(0,2):
    for j in range(0,5):
        psi.append(tensor(basis(2,i), basis(5,j)))

U = []
for i in range(0, 10):
    U_res = propagate(initial_model, q1_X_gate, psi[i], ts, 'qutip_sesolve')
    U.append(U_res)

#####



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

