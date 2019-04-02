"""C3PO configuration file"""

from numpy import pi

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

initial_model = c3po.Model(model_init)

H = initial_model.get_hamiltonian()

print(H)
"""
q1_X_gate = Gate(q, qt.sigmax())
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
