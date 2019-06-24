"""Methods to create the objects needed for GOAT propagation"""
import numpy as np


def get_initial_state(u_init, n_params):
    """
    Extend initial unitary by one per parameter
    """
    return np.kron(np.eye(n_params, 1, 0), u_init)


def get_step_matrix(h_sys, grad_list):
    """
    The GOAT Hamiltonian that contains the physical Hamiltonian and its
    gradients. It has the form:
    [ H    , 0, 0 ... 0]
    [dH_dp1, H, 0 ... 0]
    [dH_dp2, 0, H ... 0]
    [ ...             H]
    """
    n_params = len(grad_list) + 1
    goat_ham = np.kron(
            np.eye(n_params, n_params, 0) * np.eye(n_params, n_params, 0).T,
            h_sys
            )

    idx = 1
    for dh in grad_list:
        mask = np.zeros([n_params, n_params])
        mask[idx, 0] = 1
        goat_ham += np.kron(mask, dh)
        idx += 1

    return goat_ham


def select_derivative(u, n_params, pos):
    """
    Unwrap derivatives from the propagated GOAT-vector. Position 0 returns the
    regular time evolution, the k-th position return gradient to the k-th
    parameter.
    """
    P = np.kron(np.eye(1, n_params, pos), np.eye(u.shapei[1], u.shape[1], 0))
    return np.matmul(P, u)
