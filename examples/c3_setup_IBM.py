"""C3PO Setup for the IBM machine"""

from numpy import pi
import qutip as qt
import c3po

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
        initial_hilbert_space
        )

H = initial_model.get_Hamiltonian([0])

print(H)

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


"""
BSB_X_gate = Gate((q, r),
        qt.tensor(qt.sigmap(), qt.sigmap())
            + qt.tensor(qt.sigmam(), qt.sigmam())
        )
"""
