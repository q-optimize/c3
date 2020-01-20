"""Libraray of fidelity functions."""
# TODO think of how to use the fidelity functions in a cleaner way

import numpy as np
import tensorflow as tf
from c3po.tf_utils import tf_ave, tf_super, tf_ketket_fid, \
    tf_superoper_unitary_overlap, tf_unitary_overlap, evaluate_sequences, \
    tf_average_fidelity, tf_superoper_average_fidelity, tf_psi_dm, \
    tf_dm_vect, tf_dmket_fid
from c3po.qt_utils import basis, perfect_gate, perfect_cliffords, \
    cliffords_decomp


def state_transfer_infid(U_dict: dict, gate: str, proj: bool):
    U = U_dict[gate]
    if proj:
        U = U[0:2, 0:2]
    lvls = U.shape[0]
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


# def population(U_dict: dict, lvl: int, gate: str):
#     U = U_dict[gate]
#     lvls = U.shape[0]
#     psi_0 = tf.constant(basis(lvls, 0), dtype=tf.complex128)
#     psi_f = tf.constant(basis(lvls, lvl), dtype=tf.complex128)
#     psi_actual = tf.matmul(U, psi_0)
#     overlap = tf_ketket_fid(psi_f, psi_actual)
#     return overlap
#
#
# def lindbladian_population(U_dict: dict, lvl: int, gate: str):
#     U = U_dict[gate]
#     lvls = int(np.sqrt(U.shape[0]))
#     psi_0 = tf.constant(basis(lvls, 0), dtype=tf.complex128)
#     dv_0 = tf_dm_vect(tf_psi_dm(psi_0))
#     psi_f = tf.constant(basis(lvls, lvl), dtype=tf.complex128)
#     dv_actual = tf.matmul(U, dv_0)
#     overlap = tf_dmket_fid(dv_actual, psi_f)
#     return overlap


def unitary_infid(U_dict: dict, gate: str, proj: bool):
    U = U_dict[gate]
    projection = 'fulluni'
    num_gates = len(gate.split(':'))
    lvls = int(U.shape[0] ** (1/num_gates))
    fid_lvls = lvls
    if proj:
        projection = 'wzeros'
        fid_lvls = 2 * num_gates
    U_ideal = tf.constant(
                perfect_gate(lvls, gate, projection),
                dtype=tf.complex128
                )
    infid = 1 - tf_unitary_overlap(U, U_ideal, lvls=fid_lvls)
    return infid


def lindbladian_unitary_infid(U_dict: dict, gate: str, proj: bool):
    # Here we deal with the projected case differently because it's not easy
    # to select the right section of the superoper
    U = U_dict[gate]
    projection = 'fulluni'
    num_gates = len(gate.split(':'))
    lvls = int(np.sqrt(U.shape[0]) ** (1/num_gates))
    fid_lvls = lvls
    if proj:
        projection = 'wzeros'
        fid_lvls = 2 * num_gates
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
    num_gates = len(gate.split(':'))
    lvls = int(U.shape[0] ** (1/num_gates))
    fid_lvls = lvls
    if proj:
        projection = 'wzeros'
        fid_lvls = 2 * num_gates
    U_ideal = tf.constant(
                perfect_gate(lvls, gate, projection),
                dtype=tf.complex128
                )
    infid = 1 - tf_average_fidelity(U, U_ideal, lvls=fid_lvls)
    return infid


def lindbladian_average_infid(U_dict: dict, gate: str, proj: bool):
    U = U_dict[gate]
    projection = 'fulluni'
    num_gates = len(gate.split(':'))
    lvls = int(np.sqrt(U.shape[0]) ** (1/num_gates))
    fid_lvls = lvls
    if proj:
        projection = 'wzeros'
        fid_lvls = 2 * num_gates
    U_ideal = tf_super(
               tf.constant(
                    perfect_gate(lvls, gate, projection),
                    dtype=tf.complex128
                    )
               )
    infid = 1 - tf_superoper_average_fidelity(U, U_ideal, lvls=fid_lvls)
    return infid


def epc_analytical(U_dict: dict, proj: bool):
    real_cliffords = evaluate_sequences(U_dict, cliffords_decomp)
    lvls = real_cliffords[0].shape[0]
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
    real_cliffords = evaluate_sequences(U_dict, cliffords_decomp)
    lvls = int(np.sqrt(real_cliffords[0].shape[0]))
    projection = 'fulluni'
    fid_lvls = lvls
    if proj:
        projection = 'wzeros'
        fid_lvls = 2
    ideal_cliffords = perfect_cliffords(lvls, projection)
    fids = []
    for C_indx in range(24):
        C_real = real_cliffords[C_indx]
        C_ideal = tf_super(
                   tf.constant(
                        ideal_cliffords[C_indx],
                        dtype=tf.complex128
                        )
                   )
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
