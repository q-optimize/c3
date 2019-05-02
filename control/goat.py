"""Methods to create the objects needed for GOAT propagation"""

from qutip import basis, tensor


def get_initial_state(u_init, n_params):
    """
    extend initial unitary by one per parameter
    """
    return tensor(basis(n_params, 0), u_init)


def get_step_matrix(self, h_sys, dham, t, n_params, args):
    """
    The GOAT Hamiltonian that contains the physical Hamiltonian and its
    gradients.
    """
    goat_ham = tensor(basis(n_params, 0) * basis(n_params, 0).dag(), h_sys)
    for p_idx in range(1, n_params):
        dhc = dham[0]
        dh = dhc[1](t, args)[p_idx-1] * dhc[0]
        for dhc in dham[1::]:
            dh += dhc[1](t, args)[p_idx-1] * dhc[0]
        goat_ham += tensor(basis(n_params, p_idx)
                           * basis(n_params, 0).dag(), dh) \
            + tensor(basis(n_params, p_idx)
                     * basis(n_params, p_idx).dag(), h_sys)
    return goat_ham


def unwrap_derivatives():
    return 0
