"""C3PO configuration file"""

from importlib import reload


from numpy import pi
import qutip as qt
import c3po
from c3po.utils.tf_utils import *
"""
This is  disabled for now. The idea is to generalize the setup part later and use the System class
to construct a model.


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
        'qubit_1' : {'freq' : 6e9*2*pi},
        'cavity' : {'freq' : 9e9*2*pi}
        }
initial_couplings =  {
        'q1_cav' : {'strength' : 150e6*2*pi}
        }
initial_hilbert_space = {
        'qubit_1' : 2,
        'cavity' : 5
        }
model_init = [
        initial_parameters,
        initial_couplings,
        initial_hilbert_space
        ]

tf_log_level_info()

set_tf_log_level(2)

print("current level: " + str(get_tf_log_level()))

sess = tf_setup()


initial_model = c3po.Model(initial_parameters, initial_couplings, initial_hilbert_space, "True")

initial_model.set_tf_session(sess)

H = initial_model.get_Hamiltonian([0])

print(H)

q1_X_gate = c3po.Gate('qubit_1', qt.sigmax())

handmade_pulse = {'control1' : {
                      'carrier1' : {
                          'freq' : 6e9*2*pi,
                          'pulses' : { 
                            'pulse1' : {
                                'amp' : 15e6*2*pi,
                                't_up' : 5e-9,
                                't_down' : 45e-9,
                                'xy_angle' : 0
                                }
                            }
                          }
                      }
                }

q1_X_gate.set_parameters('initial', handmade_pulse)


crazy_pulse = {'control1' : {
                      'carrier1' : {
                          'freq' : 6e9*2*pi,
                          'pulses' : { 
                            'pulse1' : {
                                'amp' : 15e6*2*pi,
                                't_up' : 5e-9,
                                't_down' : 45e-9,
                                'xy_angle' : 0
                                }
                            }
                          },
                      'carrier2' : {
                          'freq' : 6e9*2*pi,
                          'pulses' : { 
                            'pulse1' : {
                                'amp' : 15e6*2*pi,
                                't_up' : 5e-9,
                                't_down' : 45e-9,
                                'xy_angle' : 0
                                },
                            'pulse2' : {
                                'amp' : 20e6*2*pi,
                                't_up' : 10e-9,
                                't_down' : 4e-9,
                                'xy_angle' : pi/2
                                }
                            }
                          }
                      }
                }


"""
BSB_X_gate = Gate((q, r),
        qt.tensor(qt.sigmap(), qt.sigmap()) + qt.tensor(qt.sigmam(), qt.sigmam())
        )

simulation_backend = GOAT('tensorflow')  # Might be standard and not even shown here

fid = Measurement('gate_fidelity_with_gradient', simulation_backend)

problem_one = Problem(WMI_memory, initial_model, fid)

problem_one.optimize_pulse(q1_X_gate)

best_params = q1_X_gate.get_open_loop('physical_scale')
best_x = q1_X_gate.get_open_loop('search_scale')
"""
