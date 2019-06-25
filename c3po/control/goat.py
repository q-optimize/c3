"""Methods to create the objects needed for GOAT propagation"""
import numpy as np

def get_initial_state(u_init, n_params):
    """
    extend initial unitary by one per parameter
    """
    return np.kron(np.eye(1, n_params, 0), u_init)


def get_step_matrix(h_sys, dham):
    """
    The GOAT Hamiltonian that contains the physical Hamiltonian and its
    gradients.
    """
    n_params = len(dham) + 1
    goat_ham = np.kron(
            np.eye(n_params, n_params, 0) * np.eye(n_params, n_params, 0).T,
            h_sys
            )
    goat_ham += np.kron(dham[0].shape[0], np.eye(1, n_params, 0))
    return goat_ham


def unwrap_derivatives():
    return 0
