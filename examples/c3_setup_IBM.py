"""C3PO Setup for the IBM machine"""

from numpy import pi
import qutip as qt
import c3po

initial_parameters = {
        'q1': {'freq': 6e9*2*pi, 'delta': 100e6 * 2 * pi},
        'r1': {'freq': 9e9*2*pi}
        }
initial_couplings = {
        ('q1', 'r1'): {'strength': 150e6*2*pi}
        }
initial_hilbert_space = {
        'q1': 2,
        'r1': 5
        }
comp_hilbert_space = {
        'q1': 2,
        'r1': 5
        }
model_types = {
        'components': {
            'q1': c3po.utils.hamiltonians.duffing,
            'r1': c3po.utils.hamiltonians.resonator},
        'couplings': {
            ('q1', 'r1'): c3po.utils.hamiltonians.int_XX},
        'drives': {
            'q1': c3po.utils.hamiltonians.drive},
        }
initial_model = c3po.Model(
        initial_parameters,
        initial_couplings,
        initial_hilbert_space,
        comp_hilbert_space=comp_hilbert_space,
        model_types=model_types,
        )

H = initial_model.get_Hamiltonian([0])

print(H)

q1_X_gate = c3po.Gate('qubit_1', qt.sigmax())

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

q1_X_gate.set_parameters('initial', handmade_pulse)
q1_X_gate.set_bounds(pulse_bounds)

# Create simulation class
def evolution():
    return 0
simulation_chip = c3po.fidelity.measurement.Simulation(initial_model,evolution)

# Create experiment class
def evaluate_sequences():
    return fidelities


