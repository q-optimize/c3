"""C3PO Setup for the IBM machine"""

from numpy import pi

import numpy as np
import qutip as qt
import matplotlib.pyplot as plt

import c3po

from c3po.main.model import Model as mdl
from c3po.fidelity.measurement import Simulation as sim
from c3po.main.gate import Gate as gt

from c3po.utils.tf_utils import *
from c3po.evolution.propagation import *

from tensorflow.python.ops.parallel_for.gradients import jacobian
import time

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
        'cavity': 2
        }
initial_model = mdl(
        initial_parameters,
        initial_couplings,
        initial_hilbert_space,
        "True"
        )

initial_model.set_tf_session(sess)

H = initial_model.get_Hamiltonian([0])

print(H)

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
                            'amp': 25e6*2*pi,
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
    basis(2,0)
).full()

U0 = tensor(
    basis(2,0),
    basis(2,0)
).full()

X_gate = gt('qubit_1', U_goal, T_final=50e-9)
pulse_bounds = {
        'control1': {
            'carrier1': {
                'pulses': {
                    'pulse': {
                        'params': {
                            'amp': [10e6*2*pi, 50e6*2*pi],
                            'freq_offset': [-1e9*2*pi, 1e9*2*pi]
                            }
                        }
                    }
                }
            }
        }

X_gate.set_parameters('initial', handmade_pulse)
X_gate.set_bounds(pulse_bounds)

# METHOD 1 breaking tensorflow
class Gateset:
    def __init__(self, X90p, gates):
        self.X90p = X90p
        self.gates = gates
        # gates = ['X90p','Y90p','X90m','Y90m']
        # even more gates = ['Xm','Xp','Ym','Yp']
        self.angles = {X90p':0,'Y90p':pi/2,'X90m':pi,'Y90m':-pi/2,}
        # more angles {'Xp':0,'Xm':pi,'Yp':pi/2,'Ym':-pi/2}
        gate_objs = {}
        for gate in gates:
            gate_objs[gate] = X90p
            #hoping -1 is the xy angle
            gate_objs[gate].parameters['initial'][-1] = X90p.parameters['initial'][-1] + self.angles[gate]
        return gate_objs


rechenknecht = sim(initial_model,[], sess)
res = 50e9
rechenknecht.resolution=res

# Broken up, deosn't work with gradient tensorflow
def get_gates(params, gateset):
    unitaries = {}
    for gate in gateset.gates:
        Ugate, params = rechenknecht.propagation(U0, gate_objs[gate], params)
        unitaries[gate] = Ugate
    return unitaries, params

def seq_evaluation(psi0, unitaries, sequences):
    fid = []
    for seq in sequences:
        psif = psi0
        psif =
        for gate in seq:
            u = gates[gate]
            psif = matmul(u,tf.expand_dims(psif,0))
        psif_density = c3po.utils.partial_trace(psif, 0)
        #TODO make tensorflowy
        fid.append(psif_density[0][0])
    return fid

def evaluate_results(sequences, results, gates):
    results = []
    psi0 = np.array(4)
    for solution in value_batch:
        gates = get_gates(solution, gates)
        surv_probs = seq_evaluation(psi0, gates, sequences)
        results.append(1 - np.mean(surv_probs))
    return results

fridge = c3po.fidelity.measurement.Experiment(eval_seq=evaluate_sequences)
# fridge.calibrate_ORBIT([q1_X_gate, q1_Y_gate])
