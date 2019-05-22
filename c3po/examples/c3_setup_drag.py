"""C3PO Setup for the IBM machine"""

from numpy import pi
import qutip as qt
import c3po

initial_parameters = {
        'qubit_1': {'freq': 6e9*2*pi, 'delta': 100e6 * 2 * pi},
        'cavity': {'freq': 9e9*2*pi}
        }
initial_couplings = {
        'q1_cav': {'strength': 150e6*2*pi}
        }
initial_hilbert_space = {
        'qubit_1': 2,
        'cavity': 5
        }
comp_hilbert_space = {
        'qubit_1': 2,
        'cavity': 5
        }
model_types = {
        'qubit_1': 'multi',  # other options: 'simple'
        'cavity': 'harmonic',
        'interaction': 'XX',   # other option 'JC', or 'JC' and 'RWA' resp.
        'drive': 'direct'  # other option 'indirect'
        }

initial_model = c3po.Model(
        initial_parameters,
        initial_couplings,
        initial_hilbert_space,
        model_types,
        )

H = initial_model.get_Hamiltonian([0])

print(H)

q1_X_gate = c3po.Gate('qubit_1', qt.sigmax(), env_shape='DRAG')

handmade_pulse = {
        'control': {
            'carrier': {
                'freq': 6e9*2*pi,
                'pulses': {
                    'pulse': {
                        'amp': 15e6*2*pi,
                        'T': 20e-9,
                        'sigma': 2e-9,
                        'xy_angle': 0,
                        'type': 'flattop'
                        },
                    'drag': {
                        'detuning': initial_parameters['qubit1']['delta'],
                        'type': 'drag',
                        'orig': 'pulse'
                        }
                    }
                }
            }
        }

q1_X_gate.set_parameters('initial', handmade_pulse)
