"""Library of fidelity functions."""
# TODO think of how to use the fidelity functions in a cleaner way

import numpy as np
import tensorflow as tf

# import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from c3.utils.tf_utils import (
    tf_ave,
    tf_super,
    tf_abs,
    tf_ketket_fid,
    tf_superoper_unitary_overlap,
    tf_unitary_overlap,
    tf_dm_to_vec,
    tf_average_fidelity,
    tf_superoper_average_fidelity,
    tf_state_to_dm,
    evaluate_sequences,
)
from c3.utils.qt_utils import (
    basis,
    perfect_gate,
    perfect_cliffords,
    cliffords_decomp,
    cliffords_decomp_xId,
    single_length_RB,
    cliffords_string,
    projector,
)

fidelities = dict()


def fid_reg_deco(func):
    """
    Decorator for making registry of functions
    """
    fidelities[str(func.__name__)] = func
    return func


@fid_reg_deco
def state_transfer_infid_set(U_dict: dict, index, dims, psi_0, proj=True):
    """
    Mean state transfer infidelity.

    Parameters
    ----------
    U_dict : dict
        Contains unitary representations of the gates, identified by a key.
    index : int
        Index of the qubit(s) in the Hilbert space to be evaluated
    dims : list
        List of dimensions of qubits
    psi_0 : tf.Tensor
        Initial state of the device
    proj : boolean
        Project to computational subspace

    Returns
    -------
    tf.float
        State infidelity, averaged over the gates in U_dict
    """
    infids = []
    for gate in U_dict.keys():
        infid = state_transfer_infid(U_dict, gate, index, dims, psi_0, proj)
        infids.append(infid)
    return tf.reduce_mean(infids)


def state_transfer_infid(U_dict: dict, gate: str, index, dims, psi_0, proj: bool):
    """
    Single gate state transfer infidelity.

    Parameters
    ----------
    U_dict : dict
        Contains unitary representations of the gates, identified by a key.
    index : int
        Index of the qubit(s) in the Hilbert space to be evaluated
    gate : str
        One of the keys of U_dict, selects the gate to be evaluated
    dims : list
        List of dimensions of qubits
    psi_0 : tf.Tensor
        Initial state of the device
    proj : boolean
        Project to computational subspace

    Returns
    -------
    tf.float
        State infidelity for the selected gate

    """
    U = U_dict[gate]
    projection = "fulluni"
    if proj:
        projection = "wzeros"
    U_ideal = tf.constant(
        perfect_gate(gate, index, dims, projection), dtype=tf.complex128
    )
    psi_ideal = tf.matmul(U_ideal, psi_0)
    psi_actual = tf.matmul(U, psi_0)
    overlap = tf_ketket_fid(psi_ideal, psi_actual)
    infid = 1 - overlap
    return infid


@fid_reg_deco
def unitary_infid(U_dict: dict, gate: str, index, dims, proj: bool):
    """
    Unitary overlap between ideal and actually performed gate.

    Parameters
    ----------
    U_dict : dict
        Contains unitary representations of the gates, identified by a key.
    index : int
        Index of the qubit(s) in the Hilbert space to be evaluated
    gate : str
        One of the keys of U_dict, selects the gate to be evaluated
    dims : list
        List of dimensions of qubits
    proj : boolean
        Project to computational subspace

    Returns
    -------
    tf.float
        Unitary fidelity.
    """
    U = U_dict[gate]
    projection = "fulluni"
    fid_lvls = np.prod([dims[i] for i in index])
    if proj:
        projection = "wzeros"
        fid_lvls = 2 ** len(index)
    U_ideal = tf.constant(
        perfect_gate(gate, index, dims, projection), dtype=tf.complex128
    )
    infid = 1 - tf_unitary_overlap(U, U_ideal, lvls=fid_lvls)
    return infid


@fid_reg_deco
def unitary_infid_set(U_dict: dict, index, dims, eval, proj=True):
    """
    Mean unitary overlap between ideal and actually performed gate for the gates in
    U_dict.

    Parameters
    ----------
    U_dict : dict
        Contains unitary representations of the gates, identified by a key.
    index : int
        Index of the qubit(s) in the Hilbert space to be evaluated
    dims : list
        List of dimensions of qubits
    proj : boolean
        Project to computational subspace

    Returns
    -------
    tf.float
        Unitary fidelity.
    """
    infids = []
    for gate in U_dict.keys():
        infid = unitary_infid(U_dict, gate, index, dims, proj)
        infids.append(infid)
    return tf.reduce_mean(infids)


@fid_reg_deco
def lindbladian_unitary_infid(U_dict: dict, gate: str, index, dims, proj: bool):
    """
    Variant of the unitary fidelity for the Lindbladian propagator.

    Parameters
    ----------
    U_dict : dict
        Contains unitary representations of the gates, identified by a key.
    index : int
        Index of the qubit(s) in the Hilbert space to be evaluated
    gate : str
        One of the keys of U_dict, selects the gate to be evaluated
    dims : list
        List of dimensions of qubits
    proj : boolean
        Project to computational subspace

    Returns
    -------
    tf.float
        Overlap fidelity for the Lindblad propagator.
    """
    # Here we deal with the projected case differently because it's not easy
    # to select the right section of the superoper
    U = U_dict[gate]
    projection = "fulluni"
    fid_lvls = np.prod([dims[i] for i in index])
    if proj:
        projection = "wzeros"
        fid_lvls = 2 ** len(index)
    U_ideal = tf_super(
        tf.constant(perfect_gate(gate, index, dims, projection), dtype=tf.complex128)
    )
    infid = 1 - tf_superoper_unitary_overlap(U, U_ideal, lvls=fid_lvls)
    return infid


@fid_reg_deco
def lindbladian_unitary_infid_set(U_dict: dict, index, dims, eval, proj=True):
    """
    Variant of the mean unitary fidelity for the Lindbladian propagator.

    Parameters
    ----------
    U_dict : dict
        Contains unitary representations of the gates, identified by a key.
    index : int
        Index of the qubit(s) in the Hilbert space to be evaluated
    dims : list
        List of dimensions of qubits
    proj : boolean
        Project to computational subspace

    Returns
    -------
    tf.float
        Mean overlap fidelity for the Lindblad propagator for all gates in U_dict.
    """
    infids = []
    for gate in U_dict.keys():
        infid = lindbladian_unitary_infid(U_dict, gate, index, dims, proj)
        infids.append(infid)
    return tf.reduce_mean(infids)


@fid_reg_deco
def average_infid_CZ(U_dict: dict, index, dims, eval, proj=True):
    """
    Average fidelity uses the Pauli basis to compare. Thus, perfect gates are
    always 2x2 (per qubit) and the actual unitary needs to be projected down.
    Variant for two-qubit gates.

    Parameters
    ----------
    U_dict : dict
        Contains unitary representations of the gates, identified by a key.
    index : int
        Index of the qubit(s) in the Hilbert space to be evaluated
    dims : list
        List of dimensions of qubits
    proj : boolean
        Project to computational subspace
    """
    proj = projector(dims, index)
    U = proj @ U_dict["Id:CZ"] @ proj.T
    subspace_dims = [dims[index[0]], dims[index[1]]]
    U_ideal = tf.constant(perfect_gate("CZ", index=[0, 1], dims=[2, 2]))
    infid = 1 - tf_average_fidelity(U, U_ideal, lvls=subspace_dims)
    return infid


@fid_reg_deco
def average_infid_simult(U_dict: dict, gate: str, index, dims, proj=True):
    """
    Average fidelity uses the Pauli basis to compare. Thus, perfect gates are
    always 2x2 (per qubit) and the actual unitary needs to be projected down.
    Variant for simultaneous single qubit gates.

    Parameters
    ----------
    U_dict : dict
        Contains unitary representations of the gates, identified by a key.
    index : int
        Index of the qubit(s) in the Hilbert space to be evaluated
    dims : list
        List of dimensions of qubits
    proj : boolean
        Project to computational subspace
    """
    proj = projector(dims, index)
    U = proj @ U_dict[gate] @ proj.T
    gate_split = gate.split(":")
    two_qubit_gate = ":".join([gate_split[index[0]], gate_split[index[1]]])
    subspace_dims = [dims[index[0]], dims[index[1]]]
    U_ideal = tf.constant(perfect_gate(two_qubit_gate, index=[0, 1], dims=[2, 2]))
    infid = 1 - tf_average_fidelity(U, U_ideal, lvls=subspace_dims)
    return infid


@fid_reg_deco
def average_infid(U_dict: dict, gate: str, index, dims, proj=True):
    """
    Average fidelity uses the Pauli basis to compare. Thus, perfect gates are
    always 2x2 (per qubit) and the actual unitary needs to be projected down.

    Parameters
    ----------
    U_dict : dict
        Contains unitary representations of the gates, identified by a key.
    index : int
        Index of the qubit(s) in the Hilbert space to be evaluated
    dims : list
        List of dimensions of qubits
    proj : boolean
        Project to computational subspace
    """
    U = U_dict[gate]
    U_ideal = tf.constant(
        perfect_gate(gate, index, dims=[2] * len(dims)), dtype=tf.complex128
    )
    infid = 1 - tf_average_fidelity(U, U_ideal, lvls=dims)
    return infid


@fid_reg_deco
def average_infid_set(U_dict: dict, index, dims, eval, proj=True):
    """
    Mean average fidelity over all gates in U_dict.

    Parameters
    ----------
    U_dict : dict
        Contains unitary representations of the gates, identified by a key.
    index : int
        Index of the qubit(s) in the Hilbert space to be evaluated
    dims : list
        List of dimensions of qubits
    proj : boolean
        Project to computational subspace

    Returns
    -------
    tf.float64
        Mean average fidelity
    """
    infids = []
    for gate in U_dict.keys():
        infid = average_infid(U_dict, gate, index, dims, proj)
        infids.append(infid)
    return tf.reduce_mean(infids)


@fid_reg_deco
def average_infid_seq(U_dict: dict, index, dims, eval, proj=True):
    """
    Average sequence fidelity over all gates in U_dict.

    Parameters
    ----------
    U_dict : dict
        Contains unitary representations of the gates, identified by a key.
    index : int
        Index of the qubit(s) in the Hilbert space to be evaluated
    dims : list
        List of dimensions of qubits
    proj : boolean
        Project to computational subspace

    Returns
    -------
    tf.float64
        Mean average fidelity
    """
    fid = 1
    for gate in U_dict.keys():
        fid *= 1 - average_infid(U_dict, gate, index, dims, proj)
    return 1 - fid


@fid_reg_deco
def lindbladian_average_infid(U_dict: dict, gate: str, index, dims, proj=True):
    """
    Average fidelity uses the Pauli basis to compare. Thus, perfect gates are
    always 2x2 (per qubit) and the actual unitary needs to be projected down.

    Parameters
    ----------
    U_dict : dict
        Contains unitary representations of the gates, identified by a key.
    index : int
        Index of the qubit(s) in the Hilbert space to be evaluated
    dims : list
        List of dimensions of qubits
    proj : boolean
        Project to computational subspace
    """
    U = U_dict[gate]
    ideal = tf.constant(
        perfect_gate(gate, index, dims=[2] * len(dims)), dtype=tf.complex128
    )
    U_ideal = tf_super(ideal)
    infid = 1 - tf_superoper_average_fidelity(U, U_ideal, lvls=dims)
    return infid


@fid_reg_deco
def lindbladian_average_infid_set(U_dict: dict, index, dims, eval, proj=True):
    """
    Mean average fidelity over all gates in U_dict.

    Parameters
    ----------
    U_dict : dict
        Contains unitary representations of the gates, identified by a key.
    index : int
        Index of the qubit(s) in the Hilbert space to be evaluated
    dims : list
        List of dimensions of qubits
    proj : boolean
        Project to computational subspace

    Returns
    -------
    tf.float64
        Mean average fidelity
    """
    infids = []
    for gate in U_dict.keys():
        infid = lindbladian_average_infid(U_dict, gate, index, dims, proj)
        infids.append(infid)
    return tf.reduce_mean(infids)


@fid_reg_deco
def epc_analytical(U_dict: dict, index, dims, proj: bool, cliffords=False):
    # TODO check this work with new index and dims (double-check)
    num_gates = len(dims)
    if cliffords:
        real_cliffords = evaluate_sequences(U_dict, [[C] for C in cliffords_string])
    elif num_gates == 1:
        real_cliffords = evaluate_sequences(U_dict, cliffords_decomp)
    elif num_gates == 2:
        real_cliffords = evaluate_sequences(U_dict, cliffords_decomp_xId)
    ideal_cliffords = perfect_cliffords(lvls=[2] * num_gates, num_gates=num_gates)
    fids = []
    for C_indx in range(24):
        C_real = real_cliffords[C_indx]
        C_ideal = tf.constant(ideal_cliffords[C_indx], dtype=tf.complex128)
        ave_fid = tf_average_fidelity(C_real, C_ideal, lvls=dims)
        fids.append(ave_fid)
    infid = 1 - tf_ave(fids)
    return infid


@fid_reg_deco
def lindbladian_epc_analytical(U_dict: dict, index, dims, proj: bool, cliffords=False):
    num_gates = len(dims)
    if cliffords:
        real_cliffords = evaluate_sequences(U_dict, [[C] for C in cliffords_string])
    elif num_gates == 1:
        real_cliffords = evaluate_sequences(U_dict, cliffords_decomp)
    elif num_gates == 2:
        real_cliffords = evaluate_sequences(U_dict, cliffords_decomp_xId)
    ideal_cliffords = perfect_cliffords(lvls=[2] * num_gates, num_gates=num_gates)
    fids = []
    for C_indx in range(24):
        C_real = real_cliffords[C_indx]
        C_ideal = tf_super(tf.constant(ideal_cliffords[C_indx], dtype=tf.complex128))
        ave_fid = tf_superoper_average_fidelity(C_real, C_ideal, lvls=dims)
        fids.append(ave_fid)
    infid = 1 - tf_ave(fids)
    return infid


@fid_reg_deco
def populations(state, lindbladian):
    if lindbladian:
        diag = []
        dim = int(np.sqrt(len(state)))
        indeces = [n * dim + n for n in range(dim)]
        for indx in indeces:
            diag.append(state[indx])
        return np.abs(diag)
    else:
        return np.abs(state) ** 2


@fid_reg_deco
def population(U_dict: dict, lvl: int, gate: str):
    U = U_dict[gate]
    lvls = U.shape[0]
    psi_0 = tf.constant(basis(lvls, 0), dtype=tf.complex128)
    psi_actual = tf.matmul(U, psi_0)
    return populations(psi_actual, lindbladian=False)[lvl]


def lindbladian_population(U_dict: dict, lvl: int, gate: str):
    U = U_dict[gate]
    lvls = int(np.sqrt(U.shape[0]))
    psi_0 = tf.constant(basis(lvls, 0), dtype=tf.complex128)
    dv_0 = tf_dm_to_vec(tf_state_to_dm(psi_0))
    dv_actual = tf.matmul(U, dv_0)
    return populations(dv_actual, lindbladian=True)[lvl]


@fid_reg_deco
def RB(
    U_dict,
    min_length: int = 5,
    max_length: int = 500,
    num_lengths: int = 20,
    num_seqs: int = 30,
    logspace=False,
    lindbladian=False,
    padding="",
):
    gate = list(U_dict.keys())[0]
    U = U_dict[gate]
    dim = int(U.shape[0])
    psi_init = tf.constant(basis(dim, 0), dtype=tf.complex128)
    if logspace:
        lengths = np.rint(
            np.logspace(np.log10(min_length), np.log10(max_length), num=num_lengths)
        ).astype(int)
    else:
        lengths = np.rint(np.linspace(min_length, max_length, num=num_lengths)).astype(
            int
        )
    surv_prob = []
    for L in lengths:
        seqs = single_length_RB(num_seqs, L, padding)
        Us = evaluate_sequences(U_dict, seqs)
        pop0s = []
        for U in Us:
            pops = populations(tf.matmul(U, psi_init), lindbladian)
            pop0s.append(float(pops[0]))
        surv_prob.append(pop0s)

    def RB_fit(len, r, A, B):
        return A * r ** (len) + B

    bounds = (0, 1)
    init_guess = [0.9, 0.5, 0.5]
    fitted = False
    while not fitted:
        try:
            means = np.mean(surv_prob, axis=1)
            stds = np.std(surv_prob, axis=1) / np.sqrt(len(surv_prob[0]))
            solution, cov = curve_fit(
                RB_fit, lengths, means, sigma=stds, bounds=bounds, p0=init_guess
            )
            r, A, B = solution
            fitted = True
        except Exception as message:
            print(message)
            if logspace:
                new_lengths = np.rint(
                    np.logspace(
                        np.log10(max_length + min_length),
                        np.log10(max_length * 2),
                        num=num_lengths,
                    )
                ).astype(int)
            else:
                new_lengths = np.rint(
                    np.linspace(
                        max_length + min_length, max_length * 2, num=num_lengths
                    )
                ).astype(int)
            max_length = max_length * 2
            for L in new_lengths:
                seqs = single_length_RB(num_seqs, L, padding)
                Us = evaluate_sequences(U_dict, seqs)
                pop0s = []
                for U in Us:
                    pops = populations(tf.matmul(U, psi_init), lindbladian)
                    pop0s.append(float(pops[0]))
                surv_prob.append(pop0s)
            lengths = np.append(lengths, new_lengths)
    epc = 0.5 * (1 - r)
    epg = 1 - ((1 - epc) ** (1 / 4))  # TODO: adjust to be mean length of
    return epg


@fid_reg_deco
def lindbladian_RB_left(
    U_dict: dict,
    gate: str,
    index,
    dims,
    proj: bool = False,
):
    return RB(U_dict, padding="left")


@fid_reg_deco
def lindbladian_RB_right(U_dict: dict, gate: str, index, dims, proj: bool):
    return RB(U_dict, padding="right")


@fid_reg_deco
def leakage_RB(
    U_dict,
    min_length: int = 5,
    max_length: int = 500,
    num_lengths: int = 20,
    num_seqs: int = 30,
    logspace=False,
    lindbladian=False,
):
    gate = list(U_dict.keys())[0]
    U = U_dict[gate]
    dim = int(U.shape[0])
    psi_init = tf.constant(basis(dim, 0), dtype=tf.complex128)
    if logspace:
        lengths = np.rint(
            np.logspace(np.log10(min_length), np.log10(max_length), num=num_lengths)
        ).astype(int)
    else:
        lengths = np.rint(np.linspace(min_length, max_length, num=num_lengths)).astype(
            int
        )
    comp_surv = []
    surv_prob = []
    for L in lengths:
        seqs = single_length_RB(num_seqs, L)
        Us = evaluate_sequences(U_dict, seqs)
        pop0s = []
        pop_comps = []
        for U in Us:
            pops = populations(tf.matmul(U, psi_init), lindbladian)
            pop0s.append(float(pops[0]))
            pop_comps.append(float(pops[0]) + float(pops[1]))
        surv_prob.append(pop0s)
        comp_surv.append(pop_comps)

    def RB_leakage(len, r_leak, A_leak, B_leak):
        return A_leak + B_leak * r_leak ** (len)

    bounds = (0, 1)
    init_guess = [0.9, 0.5, 0.5]
    fitted = False
    while not fitted:
        try:
            comp_means = np.mean(comp_surv, axis=1)
            comp_stds = np.std(comp_surv, axis=1) / np.sqrt(len(comp_surv[0]))
            solution, cov = curve_fit(
                RB_leakage,
                lengths,
                comp_means,
                sigma=comp_stds,
                bounds=bounds,
                p0=init_guess,
            )
            r_leak, A_leak, B_leak = solution
            fitted = True
        except Exception as message:
            print(message)
            if logspace:
                new_lengths = np.rint(
                    np.logspace(
                        np.log10(max_length + min_length),
                        np.log10(max_length * 2),
                        num=num_lengths,
                    )
                ).astype(int)
            else:
                new_lengths = np.rint(
                    np.linspace(
                        max_length + min_length, max_length * 2, num=num_lengths
                    )
                ).astype(int)
            max_length = max_length * 2
            for L in new_lengths:
                seqs = single_length_RB(num_seqs, L)
                Us = evaluate_sequences(U_dict, seqs)
                pop0s = []
                pop_comps = []
                for U in Us:
                    pops = populations(tf.matmul(U, psi_init), lindbladian)
                    pop0s.append(float(pops[0]))
                    pop_comps.append(float(pops[0]))
                surv_prob.append(pop0s)
                comp_surv.append(pop_comps)
            lengths = np.append(lengths, new_lengths)

    def RB_surv(len, r, A, C):
        return A + B_leak * r_leak ** (len) + C * r ** (len)

    bounds = (0, 1)
    init_guess = [0.9, 0.5, 0.5]

    fitted = False
    while not fitted:
        try:
            surv_means = np.mean(surv_prob, axis=1)
            surv_stds = np.std(surv_prob, axis=1) / np.sqrt(len(surv_prob[0]))
            solution, cov = curve_fit(
                RB_surv,
                lengths,
                surv_means,
                sigma=surv_stds,
                bounds=bounds,
                p0=init_guess,
            )
            r, A, C = solution
            fitted = True
        except Exception as message:
            print(message)
            if logspace:
                new_lengths = np.rint(
                    np.logspace(
                        np.log10(max_length + min_length),
                        np.log10(max_length * 2),
                        num=num_lengths,
                    )
                ).astype(int)
            else:
                new_lengths = np.rint(
                    np.linspace(
                        max_length + min_length, max_length * 2, num=num_lengths
                    )
                ).astype(int)
            max_length = max_length * 2
            for L in new_lengths:
                seqs = single_length_RB(num_seqs, L)
                Us = evaluate_sequences(U_dict, seqs)
                pop0s = []
                pop_comps = []
                for U in Us:
                    pops = populations(tf.matmul(U, psi_init), lindbladian)
                    pop0s.append(float(pops[0]))
                    pop_comps.append(float(pops[0]))
                surv_prob.append(pop0s)
                comp_surv.append(pop_comps)
            lengths = np.append(lengths, new_lengths)

    leakage = (1 - A_leak) * (1 - r_leak)
    seepage = A_leak * (1 - r_leak)
    fid = 0.5 * (r + 1 - leakage)
    epc = 1 - fid
    return epc, leakage, seepage, r_leak, A_leak, B_leak, r, A, C


@fid_reg_deco
def orbit_infid(
    U_dict,
    RB_number: int = 30,
    RB_length: int = 20,
    lindbladian=False,
    shots: int = None,
    seqs=None,
    noise=None,
):
    if not seqs:
        seqs = single_length_RB(RB_number=RB_number, RB_length=RB_length)
    Us = evaluate_sequences(U_dict, seqs)
    infids = []
    for U in Us:
        dim = int(U.shape[0])
        psi_init = tf.constant(basis(dim, 0), dtype=tf.complex128)
        psi_actual = tf.matmul(U, psi_init)
        pop0 = tf_abs(psi_actual[0]) ** 2
        p1 = 1 - pop0
        if shots:
            vals = tf.keras.backend.random_binomial(
                [shots],
                p=p1,
                dtype=tf.float64,
            )
            # if noise:
            #     vals = vals + (np.random.randn(shots) * noise)
            infid = tf.reduce_mean(vals)
        else:
            infid = p1
            # if noise:
            #     infid = infid + (np.random.randn() * noise)
        if noise:
            infid = infid + (np.random.randn() * noise)

        infids.append(infid)
    return tf_ave(infids)
