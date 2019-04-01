"""Methods to create the objects needed for GOAT propagation"""

def get_initial_state(self, u_init)
        return tensor(basis(self.n_params, 0), u_init)  # extend initial unitary by one per parameter

def get_step_matrix(self, h_sys, dham, t, args):
    n_params = self.n_params  # Do I need to do this?
    goat_ham = tensor(basis(n_params, 0) * basis(n_params, 0).dag(), h)
    for p_idx in range(1, n_params):  # add one line in the goat hamiltonian per parameter
        for dhc in dham:
            dh += dhc[1](t, args)[p_idx-1] * dhc[0]
        goat_ham += tensor(basis(n_params, p_idx) * basis(n_params, 0).dag(), dh) \
                + tensor(basis(n_params, p_idx) * basis(n_params, p_idx).dag(), h)
    return goat_ham

def unwrap_derivatives()
