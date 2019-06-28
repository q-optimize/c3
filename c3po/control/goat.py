"""
Methods to create the objects needed for GOAT propagation
TODO: Investigate using sparse matrices in the creation.
"""
import numpy as np

def get_initial_state(u_init, n_params):
    """
    Extend initial unitary by one per parameter
    """
    return np.kron(np.eye(n_params+1, 1, 0), u_init)


def get_step_matrix(h, grads):
    """
    The GOAT Hamiltonian that contains the physical Hamiltonian and its
    gradients. It has the form:
    [ H    , 0, 0 ... 0]\n
    [dH_dp1, H, 0 ... 0]\n
    [dH_dp2, 0, H ... 0]\n
    [ ...             H]
    """
    n_params = len(grads) + 1
    goat_ham = np.kron(
            np.eye(n_params, n_params, 0) * np.eye(n_params, n_params, 0).T,
            h
            )

    idx = 1
    for dh in grads:
        mask = np.zeros([n_params, n_params])
        mask[idx, 0] = 1
        goat_ham += np.kron(mask, dh)
        idx += 1

    return goat_ham


def select_derivative(u, n_params, pos):
    """
    Unwrap derivatives from the propagated GOAT-vector. Position 0 returns
    the regular time evolution, the k-th position return gradient to the k-th
    parameter.
    """
    sub_dim = u.shape[0]//(n_params+1)
    P = np.kron(np.eye(1, n_params+1, pos), np.eye(sub_dim, sub_dim, 0))
    return np.matmul(P, u)
