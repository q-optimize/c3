"""Libraray of fidelity functions."""
# TODO think of how to use the fidelity functions in a cleaner way


import tensorflow as tf
from c3po.tf_utils import tf_ave, tf_super, tf_ketket_fid, \
    tf_superoper_unitary_overlap, tf_unitary_overlap, evaluate_sequences, \
    tf_average_fidelity, tf_superoper_average_fidelity
from c3po.qt_utils import basis, perfect_gate, perfect_cliffords, \
    clifford_decomp


def state_transfer_infid(U_dict: dict, gate: str, proj: bool):
    U = U_dict['X90p']
    if proj:
        U = U[0:2, 0:2]
    lvls = tf.cast(U.shape[0], U.dtype)
    U_ideal = tf.constant(
                perfect_gate(lvls, gate, proj),
                dtype=tf.complex128
                )
    psi_0 = tf.constant(basis(lvls, 0), dtype=tf.complex128)
    psi_ideal = tf.matmul(U_ideal, psi_0)
    psi_actual = tf.matmul(U, psi_0)
    overlap = tf_ketket_fid(psi_ideal, psi_actual)
    infid = 1 - overlap
    return infid


def unitary_infid(U_dict: dict, gate: str, proj: bool):
    U = U_dict[gate]
    projection = 'fulluni'
    if proj:
        U = U[0:2, 0:2]
        projection = 'compsub'
    lvls = tf.cast(U.shape[0], U.dtype)
    U_ideal = tf.constant(
                perfect_gate(lvls, gate, projection),
                dtype=tf.complex128
                )
    infid = 1 - tf_unitary_overlap(U, U_ideal)
    return infid


def lindbladian_unitary_infid(U_dict: dict, gate: str, proj: bool):
    # Here we deal with the projected case differently because it's not easy
    # to select the right section of the superoper
    U = U_dict[gate]
    projection = 'fulluni'
    lvls = tf.sqrt(tf.cast(U.shape[0], U.dtype))
    fid_lvls = lvls
    if proj:
        projection = 'wzeros'
        fid_lvls = 2
    U_ideal = tf_super(
               tf.constant(
                    perfect_gate(lvls, gate, projection),
                    dtype=tf.complex128
                    )
               )
    infid = 1 - tf_superoper_unitary_overlap(U, U_ideal, lvls=fid_lvls)
    return infid


def average_infid(U_dict: dict, gate: str, proj: bool):
    U = U_dict[gate]
    projection = 'fulluni'
    if proj:
        U = U[0:2, 0:2]
        projection = 'compsub'
    lvls = tf.cast(U.shape[0], U.dtype)
    U_ideal = tf.constant(
                perfect_gate(lvls, gate, projection),
                dtype=tf.complex128
                )
    infid = 1 - tf_average_fidelity(U, U_ideal)
    return infid


def lindbladian_average_infid(U_dict: dict, gate: str, proj: bool):
    U = U_dict[gate]
    projection = 'fulluni'
    lvls = tf.sqrt(tf.cast(U.shape[0], U.dtype))
    fid_lvls = lvls
    if proj:
        projection = 'wzeros'
        fid_lvls = 2
    U_ideal = tf.constant(
                perfect_gate(lvls, gate, projection),
                dtype=tf.complex128
                )
    infid = 1 - tf_superoper_average_fidelity(U, U_ideal, lvls=fid_lvls)
    return infid


def epc_analytical(U_dict: dict, proj: bool):
    real_cliffords = evaluate_sequences(U_dict, clifford_decomp)
    lvls = tf.cast(real_cliffords[0].shape[0], real_cliffords[0].dtype)
    projection = 'fulluni'
    if proj:
        projection = 'compsub'
        lvls = 2
    ideal_cliffords = perfect_cliffords(lvls, projection)
    fids = []
    for C_indx in range(24):
        C_real = real_cliffords[C_indx]
        if proj:
            C_real = C_real[0:2, 0:2]
        C_ideal = ideal_cliffords[C_indx]
        ave_fid = tf_average_fidelity(C_real, C_ideal)
        fids.append(ave_fid)
    infid = 1 - tf_ave(fids)
    return infid


def lindbladian_epc_analytical(U_dict: dict, proj: bool):
    real_cliffords = evaluate_sequences(U_dict, clifford_decomp)
    lvls = tf.cast(real_cliffords[0].shape[0], real_cliffords[0].dtype)
    projection = 'fulluni'
    fid_lvls = lvls
    if proj:
        projection = 'wzeros'
        fid_lvls = 2
    ideal_cliffords = perfect_cliffords(lvls, projection)
    fids = []
    for C_indx in range(24):
        C_real = real_cliffords[C_indx]
        C_ideal = ideal_cliffords[C_indx]
        ave_fid = tf_superoper_average_fidelity(C_real, C_ideal, lvls=fid_lvls)
        fids.append(ave_fid)
    infid = 1 - tf_ave(fids)
    return infid

# def orbit_infid(U_dict: dict):
#     seqs = single_length_RB(RB_number=25, RB_length=20)
#     U_seqs = evaluate_sequences(U_dict, seqs)
#     lvls = tf.cast(U_seqs[0].shape[0], U_seqs[0].dtype)
#     psi_0 = tf.constant(basis(lvls, 0), dtype=tf.complex128)
#     psi_yp = tf.constant(xy_basis(lvls, 'yp'), dtype=tf.complex128)
#     infids = []
#     for U in U_seqs:
#         psi_actual = tf.matmul(U, psi_0)
#         overlap = tf_ketket_fid(psi_yp, psi_actual)
#         infids.append(1 - overlap)
#     return tf_ave(infids)
