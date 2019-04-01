from IPython import display
import oc_backend as oc
from qutip import *
import numpy as np
import os

# Define system parameters
omega_r = 9e9 * 2 * np.pi
omega_q = 6e9 * 2 * np.pi
w_d = 37407165859.10755
delta = 100e6 * 2 * np.pi
g = 200e6 * 2 * np.pi
hbar = 1

# Define Hilbert space
N_r = 3
N_q = 4
a = tensor(destroy(N_r), qeye(N_q))
b = tensor(qeye(N_r), destroy(N_q))
qubit_control = b.dag() + b
H0 = oc.jc_hamiltonian(omega_r, a, omega_q, delta, b, g)
psi0 = tensor(basis(2,0),basis(2,0))


current_dir = os.getcwd()
br = oc.SimCMARunner(H0, psi0, w_d)
# ebr = oc.ExpCMARunner('testcmaes', base_dir, SQORE_DIR, run_sqore='run_sqore_precalh')

initial_parameters = {'Q1':
                          {'X90p': {'amp': 1, 'delta': 1, 'freq_offset': 1},
                           'Y90p': {'mapto': 'X90p'}}}

error_bars = {'Q1':
                  {'X90p': {'amp': 0.02, 'delta': 0.01, 'freq_offset': 1000}, 'Y90p': {'mapto': 'X90p'}}}

oc.cmaes_par_search(initial_parameters, error_bars, br, rep=10, rb_len=4, pop_size=5)

os.chdir(current_dir)