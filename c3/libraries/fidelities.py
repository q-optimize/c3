"""Library of fidelity functions."""

import numpy as np
import tensorflow as tf
from typing import List, Dict

from scipy.optimize import curve_fit

from c3.signal.gates import Instruction
from c3.utils.tf_utils import (
    tf_ave,
    tf_super,
    tf_abs,
    tf_ketket_fid,
    tf_superoper_unitary_overlap,
    tf_unitary_overlap,
    tf_project_to_comp,
    tf_dm_to_vec,
    tf_average_fidelity,
    tf_superoper_average_fidelity,
    tf_state_to_dm,
    evaluate_sequences,
)
from c3.utils.qt_utils import (
    basis,
    perfect_cliffords,
    cliffords_decomp,
    cliffords_decomp_xId,
    single_length_RB,
    cliffords_string,
)

fidelities = dict()


def fid_reg_deco(func):
    """
    Decorator for making registry of functions
    """
    fidelities[str(func.__name__)] = func
    return func


@fid_reg_deco
def state_transfer_infid_set(
    propagators: dict, instructions: dict, index, dims, psi_0, proj=True
):
    """
    Mean state transfer infidelity.

    Parameters
    ----------
    propagators : dict
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
        State infidelity, averaged over the gates in propagators
    """
    infids = []
    for gate, propagator in propagators.items():
        perfect_gate = instructions[gate].get_ideal_gate(dims)
        infid = state_transfer_infid(perfect_gate, propagator, index, dims, psi_0)
        infids.append(infid)
    return tf.reduce_mean(infids)


@fid_reg_deco
def state_transfer_infid(ideal: np.array, actual: tf.constant, index, dims, psi_0):
    """
    Single gate state transfer infidelity. The dimensions of psi_0 and ideal need to be
    compatible and index and dims need to project actual to these same dimensions.

    Parameters
    ----------
    ideal: np.array
        Contains ideal unitary representations of the gate
    actual: tf.Tensor
        Contains actual unitary representations of the gate
    index : int
        Index of the qubit(s) in the Hilbert space to be evaluated
    dims : list
        List of dimensions of qubits
    psi_0: tf.Tensor
        Initial state

    Returns
    -------
    tf.float
        State infidelity for the selected gate

    """
    actual_comp = tf_project_to_comp(actual, dims=dims, index=index)
    psi_ideal = tf.matmul(ideal, psi_0)
    psi_actual = tf.matmul(actual_comp, psi_0)
    overlap = tf_ketket_fid(psi_ideal, psi_actual)
    infid = 1 - overlap
    return infid


@fid_reg_deco
def unitary_infid(
    ideal: np.array, actual: tf.Tensor, index: List[int] = None, dims=None
) -> tf.Tensor:
    """
    Unitary overlap between ideal and actually performed gate.

    Parameters
    ----------
    ideal : np.array
        Ideal or goal unitary representation of the gate.
    actual : np.array
        Actual, physical unitary representation of the gate.
    index : List[int]
        Index of the qubit(s) in the Hilbert space to be evaluated
    gate : str
        One of the keys of propagators, selects the gate to be evaluated
    dims : list
        List of dimensions of qubits

    Returns
    -------
    tf.float
        Unitary fidelity.
    """
    if index is None:
        index = list(range(len(dims)))
    actual_comp = tf_project_to_comp(actual, dims=dims, index=index)
    fid_lvls = 2 ** len(index)
    infid = 1 - tf_unitary_overlap(actual_comp, ideal, lvls=fid_lvls)
    return infid


@fid_reg_deco
def unitary_infid_set(propagators: dict, instructions: dict, index, dims, n_eval=-1):
    """
    Mean unitary overlap between ideal and actually performed gate for the gates in
    propagators.

    Parameters
    ----------
    propagators : dict
        Contains actual unitary representations of the gates, resulting from physical
        simulation
    instructions : dict
        Contains the perfect unitary representations of the gates, identified by a key.
    index : List[int]
        Index of the qubit(s) in the Hilbert space to be evaluated
    dims : list
        List of dimensions of qubits
    n_eval : int
        Number of evaluation

    Returns
    -------
    tf.float
        Unitary fidelity.
    """
    infids = []
    for gate, propagator in propagators.items():
        perfect_gate = instructions[gate].get_ideal_gate(dims, index)
        infid = unitary_infid(perfect_gate, propagator, index, dims)
        infids.append(infid)
    return tf.reduce_mean(infids)


@fid_reg_deco
def lindbladian_unitary_infid(
    ideal: np.array, actual: tf.constant, index=[0], dims=[2]
) -> tf.constant:
    """
    Variant of the unitary fidelity for the Lindbladian propagator.

    Parameters
    ----------
    ideal: np.array
        Contains ideal unitary representations of the gate
    actual: tf.Tensor
        Contains actual unitary representations of the gate
    index : List[int]
        Index of the qubit(s) in the Hilbert space to be evaluated
    dims : list
        List of dimensions of qubits

    Returns
    -------
    tf.float
        Overlap fidelity for the Lindblad propagator.
    """
    U_ideal = tf_super(ideal)
    actual_comp = tf_project_to_comp(actual, dims=dims, index=index, to_super=True)
    fid_lvls = 2 ** len(index)
    infid = 1 - tf_superoper_unitary_overlap(actual_comp, U_ideal, lvls=fid_lvls)
    return infid


@fid_reg_deco
def lindbladian_unitary_infid_set(
    propagators: dict, instructions: Dict[str, Instruction], index, dims, n_eval
):
    """
    Variant of the mean unitary fidelity for the Lindbladian propagator.

    Parameters
    ----------
    propagators : dict
        Contains actual unitary representations of the gates, resulting from physical
        simulation
    instructions : dict
        Contains the perfect unitary representations of the gates, identified by a key.
    index : List[int]
        Index of the qubit(s) in the Hilbert space to be evaluated
    dims : list
        List of dimensions of qubits
    n_eval : int
        Number of evaluation

    Returns
    -------
    tf.float
        Mean overlap fidelity for the Lindblad propagator for all gates in propagators.
    """
    infids = []
    for gate, propagator in propagators.items():
        perfect_gate = instructions[gate].get_ideal_gate(dims)
        infid = lindbladian_unitary_infid(perfect_gate, propagator, index, dims)
        infids.append(infid)
    return tf.reduce_mean(infids)


@fid_reg_deco
def average_infid(
    ideal: np.array, actual: tf.Tensor, index: List[int] = [0], dims=[2]
) -> tf.constant:
    """
    Average fidelity uses the Pauli basis to compare. Thus, perfect gates are
    always 2x2 (per qubit) and the actual unitary needs to be projected down.

    Parameters
    ----------
    ideal: np.array
        Contains ideal unitary representations of the gate
    actual: tf.Tensor
        Contains actual unitary representations of the gate
    index : List[int]
        Index of the qubit(s) in the Hilbert space to be evaluated
    dims : list
        List of dimensions of qubits
    """
    actual_comp = tf_project_to_comp(actual, dims=dims, index=index)
    fid_lvls = [2] * len(index)
    infid = 1 - tf_average_fidelity(actual_comp, ideal, lvls=fid_lvls)
    return infid


@fid_reg_deco
def average_infid_set(
    propagators: dict, instructions: dict, index: List[int], dims, n_eval=-1
):
    """
    Mean average fidelity over all gates in propagators.

    Parameters
    ----------
    propagators : dict
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
    for gate, propagator in propagators.items():
        perfect_gate = instructions[gate].get_ideal_gate(dims, index)
        infid = average_infid(perfect_gate, propagator, index, dims)
        infids.append(infid)
    return tf.reduce_mean(infids)


@fid_reg_deco
def average_infid_seq(propagators: dict, instructions: dict, index, dims, n_eval=-1):
    """
    Average sequence fidelity over all gates in propagators.

    Parameters
    ----------
    propagators : dict
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
    for gate, propagator in propagators.items():
        perfect_gate = instructions[gate].get_ideal_gate(dims)
        fid *= 1 - average_infid(perfect_gate, propagator, index, dims)
    return 1 - fid


@fid_reg_deco
def lindbladian_average_infid(
    ideal: np.array, actual: tf.constant, index=[0], dims=[2]
) -> tf.constant:
    """
    Average fidelity uses the Pauli basis to compare. Thus, perfect gates are
    always 2x2 (per qubit) and the actual unitary needs to be projected down.

    Parameters
    ----------
    ideal: np.array
        Contains ideal unitary representations of the gate
    actual: tf.Tensor
        Contains actual unitary representations of the gate
    index : int
        Index of the qubit(s) in the Hilbert space to be evaluated
    dims : list
        List of dimensions of qubits
    """
    U_ideal = tf_super(ideal)
    actual_comp = tf_project_to_comp(actual, dims=dims, index=index, to_super=True)
    infid = 1 - tf_superoper_average_fidelity(actual_comp, U_ideal, lvls=dims)
    return infid


@fid_reg_deco
def lindbladian_average_infid_set(
    propagators: dict, instructions: Dict[str, Instruction], index, dims, n_eval
):
    """
    Mean average fidelity over all gates in propagators.

    Parameters
    ----------
    propagators : dict
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
    for gate, propagator in propagators.items():
        perfect_gate = instructions[gate].get_ideal_gate(dims)
        infid = lindbladian_average_infid(perfect_gate, propagator, index, dims)
        infids.append(infid)
    return tf.reduce_mean(infids)


@fid_reg_deco
def epc_analytical(propagators: dict, index, dims, proj: bool, cliffords=False):
    # TODO check this work with new index and dims (double-check)
    num_gates = len(dims)
    if cliffords:
        real_cliffords = evaluate_sequences(
            propagators, [[C] for C in cliffords_string]
        )
    elif num_gates == 1:
        real_cliffords = evaluate_sequences(propagators, cliffords_decomp)
    elif num_gates == 2:
        real_cliffords = evaluate_sequences(propagators, cliffords_decomp_xId)
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
def lindbladian_epc_analytical(
    propagators: dict, index, dims, proj: bool, cliffords=False
):
    num_gates = len(dims)
    if cliffords:
        real_cliffords = evaluate_sequences(
            propagators, [[C] for C in cliffords_string]
        )
    elif num_gates == 1:
        real_cliffords = evaluate_sequences(propagators, cliffords_decomp)
    elif num_gates == 2:
        real_cliffords = evaluate_sequences(propagators, cliffords_decomp_xId)
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
def population(propagators: dict, lvl: int, gate: str):
    U = propagators[gate]
    lvls = U.shape[0]
    psi_0 = tf.constant(basis(lvls, 0), dtype=tf.complex128)
    psi_actual = tf.matmul(U, psi_0)
    return populations(psi_actual, lindbladian=False)[lvl]


def lindbladian_population(propagators: dict, lvl: int, gate: str):
    U = propagators[gate]
    lvls = int(np.sqrt(U.shape[0]))
    psi_0 = tf.constant(basis(lvls, 0), dtype=tf.complex128)
    dv_0 = tf_dm_to_vec(tf_state_to_dm(psi_0))
    dv_actual = tf.matmul(U, dv_0)
    return populations(dv_actual, lindbladian=True)[lvl]


@fid_reg_deco
def RB(
    propagators,
    min_length: int = 5,
    max_length: int = 500,
    num_lengths: int = 20,
    num_seqs: int = 30,
    logspace=False,
    lindbladian=False,
    padding="",
):
    gate = list(propagators.keys())[0]
    U = propagators[gate]
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
        Us = evaluate_sequences(propagators, seqs)
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
                Us = evaluate_sequences(propagators, seqs)
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
    propagators: dict,
    gate: str,
    index,
    dims,
    proj: bool = False,
):
    return RB(propagators, padding="left")


@fid_reg_deco
def lindbladian_RB_right(propagators: dict, gate: str, index, dims, proj: bool):
    return RB(propagators, padding="right")


@fid_reg_deco
def leakage_RB(
    propagators,
    min_length: int = 5,
    max_length: int = 500,
    num_lengths: int = 20,
    num_seqs: int = 30,
    logspace=False,
    lindbladian=False,
):
    gate = list(propagators.keys())[0]
    U = propagators[gate]
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
        Us = evaluate_sequences(propagators, seqs)
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
                Us = evaluate_sequences(propagators, seqs)
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
                Us = evaluate_sequences(propagators, seqs)
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
    propagators,
    RB_number: int = 30,
    RB_length: int = 20,
    lindbladian=False,
    shots: int = None,
    seqs=None,
    noise=None,
):
    if not seqs:
        seqs = single_length_RB(RB_number=RB_number, RB_length=RB_length)
    Us = evaluate_sequences(propagators, seqs)
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
