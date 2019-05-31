"""C3PO Setup for the IBM machine"""

from numpy import np
import qutip as qt
import c3po

initial_parameters = {
        'q1': {'freq': 6e9*2*np.pi, 'delta': 100e6 * 2 * np.pi},
        'r1': {'freq': 9e9*2*np.pi}
        }
initial_couplings = {
        ('q1', 'r1'): {'strength': 150e6*2*np.pi}
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
q1_Y_gate = c3po.Gate('qubit_2', qt.sigmay())

handmade_pulse = {
        'control1': {
            'carrier1': {
                'freq': 6e9*2*np.pi,
                'target': 'q1',  # add here?
                'pulses': {
                    'pulse1': {
                        'amp': 15e6*2*np.pi,
                        't_up': 5e-9,
                        't_down': 45e-9,
                        'xy_angle': 0,
                        'type': 'gaussian'
                        },
                    'drag': {
                        'detuning': initial_parameters['qubit1']['delta'],
                        'type': 'drag',
                        'orig': 'pulse1'
                        }
                    }
                }
            }
        }

pulse_bounds = {
        'control1': {
            'carrier1': {
                'freq': [1e9*2*np.pi, 15e9*2*np.pi],
                'pulses': {
                    'pulse1': {
                        'amp':  [1e3*2*np.pi, 10e9*2*np.pi],
                        't_up': [2e-9, 98e-9],
                        't_down': [2e-9, 98e-9],
                        'xy_angle': [-np.pi, np.pi]
                        }
                    }
                }
            }
        }

q1_X_gate.set_parameters('initial', handmade_pulse)
q1_X_gate.set_bounds(pulse_bounds)
q1_Y_gate.set_parameters('initial', map_to=q1_X_gate, xy_angle=np.pi)


# Create simulation class
def evolution():
    return 0


simulation_chip = c3po.fidelity.measurement.Simulation(initial_model, evolution)


# Create experiment class
def get_gates(solution, gates):
    unitaries = []
    for indx in range(len(solution)):
        unitaries.append(simulation_chip.evolution(solution[indx], gates[indx]))
    return unitaries


def seq_evaluation(psi0, gates, sequences):
    fid = []
    for seq in sequences:
        psif = psi0
        for gate in seq:
            u = gates[gate]
            psif = u * psif
        psif_density = c3po.utils.partial_trace(psif, 0)
        fid.append(abs(psif_density[0][0][0]))
    return fid


def evaluate_sequences(sequences, value_batch, gates):
    results = []
    psi0 = 0
    for solution in value_batch:
        gates = get_gates(solution, gates)
        surv_probs = seq_evaluation(psi0, gates, sequences)
        results.append(1 - np.mean(surv_probs))
    return results


fridge = c3po.fidelity.measurement.Experiment(eval_seq=evaluate_sequences)
fridge.calibrate_ORBIT([q1_X_gate,q1_Y_gate])
