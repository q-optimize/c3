"""Useful functions to get basis vectors and matrices of the right size."""

import numpy as np
from typing import List
from scipy.linalg import block_diag as scipy_block_diag
from scipy.linalg import expm
from c3.libraries.constants import Id, X, Y, Z, PAULIS, CLIFFORDS


def pauli_basis(dims=[2]):
    """
    Qutip implementation of the Pauli basis.

    Parameters
    ----------
    dims : list
        List of dimensions of each subspace.

    Returns
    -------
    np.array
        A square matrix containing the Pauli basis of the product space
    """

    def expand_dims(op, dim):
        """
        pad operator with zeros to be of dimension dim
        Attention! Not related to the TensorFlow function
        """
        op_out = np.zeros([dim, dim], dtype=op.dtype)
        op_out[: op.shape[0], : op.shape[1]] = op
        return op_out

    _SINGLE_QUBIT_PAULI_BASIS = (Id, X, Y, Z)
    paulis = []
    for dim in dims:
        paulis.append([expand_dims(P, dim) for P in _SINGLE_QUBIT_PAULI_BASIS])
    result = [[]]
    res_tuple = []
    # TAKEN FROM ITERTOOLS
    for pauli_set in paulis:
        result = [x + [y] for x in result for y in pauli_set]
    for prod in result:
        res_tuple.append(tuple(prod))

    # TAKEN FROM QUTIP
    size = np.prod(np.array(dims) ** 2)
    B = np.zeros((size, size), dtype=complex)
    for idx, op_tuple in enumerate(res_tuple):
        op = np_kron_n(op_tuple)
        vec = np.reshape(np.transpose(op), [-1, 1])
        B[:, idx] = vec.T.conj()
    return B


# MATH HELPERS
def np_kron_n(mat_list):
    """
    Apply Kronecker product to a list of matrices.
    """
    tmp = np.eye(1)
    for m in mat_list:
        tmp = np.kron(tmp, m)
    return tmp


def hilbert_space_kron(op, indx, dims):
    """
    Extend an operator op to the full product hilbert space
    given by dimensions in dims.

    Parameters
    ----------
    op : np.array
        Operator to be extended.
    indx : int
        Position of which subspace to extend.
    dims : list
        New dimensions of the subspace.

    Returns
    -------
    np.array
        Extended operator.
    """
    op_list = []
    for indy in range(len(dims)):
        qI = np.identity(dims[indy])
        if indy == indx:
            op_list.append(op)
        else:
            op_list.append(qI)
    if indx > len(dims) - 1:
        raise Warning(f"Index {indx} is outside the Hilbert space dimensions {dims}. ")
    return np_kron_n(op_list)


def rotation(phase: float, xyz: np.array) -> np.array:
    """General Rotation using Euler's formula.

    Parameters
    ----------
    phase : np.float
        Rotation angle.
    xyz : np.array
        Normal vector of the rotation axis.

    Returns
    -------
    np.array
        Unitary matrix
    """
    rot = np.cos(phase / 2) * Id - 1j * np.sin(phase / 2) * (
        xyz[0] * X + xyz[1] * Y + xyz[2] * Z
    )
    return rot


def basis(lvls: int, pop_lvl: int) -> np.array:
    """
    Construct a basis state vector.

    Parameters
    ----------
    lvls : int
        Dimension of the state.
    pop_lvl : int
        The populated entry.

    Returns
    -------
    np.array
        A normalized state vector with one populated entry.
    """
    psi = np.zeros([lvls, 1])
    psi[pop_lvl] = 1
    return psi


def xy_basis(lvls: int, vect: str):
    """
    Construct basis states on the X, Y and Z axis.

    Parameters
    ----------
    lvls : int
        Dimensions of the Hilbert space.
    vect : str
        Identifier of the state.
        Options:
            'zp', 'zm', 'xp', 'xm', 'yp', 'ym'

    Returns
    -------
    np.array
        A state on one of the axis of the Bloch sphere.
    """

    psi_g = basis(lvls, 0)
    psi_e = basis(lvls, 1)
    basis_states = {
        "zm": psi_g,
        "zp": psi_e,
        "xp": (psi_g + psi_e) / np.sqrt(2),
        "xm": (psi_g - psi_e) / np.sqrt(2),
        "yp": (psi_g + 1.0j * psi_e) / np.sqrt(2),
        "ym": (psi_g - 1.0j * psi_e) / np.sqrt(2),
    }
    try:
        psi = basis_states[vect]
    except KeyError:
        print("vect must be one of 'zp' 'zm' 'xp' 'xm' 'yp' 'ym'")
        psi = None
    return psi


def projector(dims, indices, outdims=None):
    """
    Computes the projector to cut down a matrix to the computational space. The
    subspaces indicated in indeces will be projected to the lowest two states,
    the rest is projected onto the lowest state. If outdims is defined projection will be performed to those states.
    """
    if outdims is None:
        outdims = [2] * len(dims)
    ids = []
    for index, dim in enumerate(dims):
        outdim = outdims[index]
        if index in indices:
            ids.append(np.eye(dim)[:outdim])
        else:
            mask = np.zeros(dim)
            mask[0] = 1
            ids.append(mask)
    return np_kron_n(ids)


def kron_ids(dims, indices, matrices):
    """
    Kronecker product of matrices at specified indices with identities everywhere else.
    """
    ids = []
    for index, dim in enumerate(dims):
        ids.append(np.eye(dim))
    for index, matrix in enumerate(matrices):
        ids[indices[index]] = matrix
    return np_kron_n(ids)


def pad_matrix(matrix, dim, padding):
    """
    Fills matrix dimensions with zeros or identity.
    """
    if padding == "compsub":
        return matrix
    elif padding == "wzeros":
        zeros = np.zeros([dim, dim])
        matrix = scipy_block_diag(matrix, zeros)
    elif padding == "fulluni":
        identity = np.eye(dim)
        matrix = scipy_block_diag(matrix, identity)
    return matrix


# NOTE: Removed perfect_gate() as in commit 0f7cba3, replaced by explicit constants in
# c3/libraries/constants.py


def perfect_parametric_gate(paulis_str, ang, dims):
    """
    Construct an ideal parametric gate.

    Parameters
    ----------
    paulis_str : str
        Names for the Pauli matrices that identify the rotation axis. Example:
            - "X" for a single-qubit rotation about the X axis
            - "Z:X" for an entangling rotation about Z on the first and X on the second qubit
    ang : float
        Angle of the rotation
    dims : list
        Dimensions of the subspaces.

    Returns
    -------
    np.array
        Ideal gate.
    """
    ps = []
    p_list = paulis_str.split(":")
    for idx, key in enumerate(p_list):
        if key not in PAULIS:
            raise KeyError(
                f"Incorrect pauli matrix {key} in position {idx}.\
                Select from {PAULIS.keys()}."
            )
        ps.append(pad_matrix(PAULIS[key], dims[idx] - 2, "wzeros"))
    gen = np_kron_n(ps)
    return expm(-1.0j / 2 * ang * gen)


def perfect_single_q_parametric_gate(pauli_str, target, ang, dims):
    """
    Construct an ideal parametric gate.

    Parameters
    ----------
    paulis_str : str
        Name for the Pauli matrices that identify the rotation axis. Example:
            - "X" for a single-qubit rotation about the X axis
    ang : float
        Angle of the rotation
    dims : list
        Dimensions of the subspaces.

    Returns
    -------
    np.array
        Ideal gate.
    """
    ps = []
    p_list = ["Id"] * len(dims)
    p_list[target] = pauli_str
    for idx, key in enumerate(p_list):
        if key not in PAULIS:
            raise KeyError(
                f"Incorrect pauli matrix {key} in position {idx}.\
                Select from {PAULIS.keys()}."
            )
        ps.append(pad_matrix(PAULIS[key], dims[idx] - 2, "wzeros"))
    gen = np_kron_n(ps)
    return expm(-1.0j / 2 * ang * gen)


def two_qubit_gate_tomography(gate):
    """
    Sequences to generate tomography for evaluating a two qubit gate.
    """
    # THE 4 GATES
    base = [["Id", "Id"], ["rx90p", "Id"], ["ry90p", "Id"], ["rx90p", "rx90p"]]
    base2 = []
    for x in base:
        for y in base:
            g = []
            for indx in range(2):
                g.append(x[indx] + ":" + y[indx])
            base2.append(g)

    S = []
    for x in base2:
        for y in base2:
            g = []
            for g1 in x:
                g.append(g1)
            g.append(gate)
            for g2 in y:
                g.append(g2)
            S.append(g)
    return S


def T1_sequence(length, target):
    """
    Generate a gate sequence to measure relaxation time in a two-qubit chip.

    Parameters
    ----------
    length : int
        Number of Identity gates.
    target : int
        Which qubit is measured.

    Returns
    -------
    list
        Relaxation sequence.

    """
    wait = ["Id"]
    prepare_1 = [f"rx90p[{str(target)}]"] * 2
    S = []
    S.extend(prepare_1)
    S.extend(wait * length)
    return S


def ramsey_sequence(length, target):
    """
    Generate a gate sequence to measure dephasing time in a two-qubit chip.

    Parameters
    ----------
    length : int
        Number of Identity gates.
    target : str
        Which qubit is measured. Options: "left" or "right"

    Returns
    -------
    list
        Dephasing sequence.

    """
    wait = ["id"]
    rotate_90 = [f"rx90p[{str(target)}]"]
    S = []
    S.extend(rotate_90)
    S.extend(wait * length)
    S.extend(rotate_90)
    return S


def ramsey_echo_sequence(length, target):
    """
    Generate a gate sequence to measure dephasing time in a two-qubit chip including a
    flip in the middle.
    This echo reduce effects detrimental to the dephasing measurement.

    Parameters
    ----------
    length : int
        Number of Identity gates. Should be even.
    target : str
        Which qubit is measured. Options: "left" or "right"

    Returns
    -------
    list
        Dephasing sequence.

    """
    wait = ["id"]
    hlength = length // 2
    rotate_90_p = [f"rx90p[{str(target)}]"]
    rotate_90_m = [f"rx90m[{str(target)}]"]
    S = []
    S.extend(rotate_90_p)
    S.extend(wait * hlength)
    S.extend(rotate_90_p)
    S.extend(rotate_90_p)
    S.extend(wait * hlength)
    S.extend(rotate_90_m)
    return S


def single_length_RB(
    RB_number: int, RB_length: int, target: int = 0
) -> List[List[str]]:
    """Given a length and number of repetitions it compiles Randomized Benchmarking
    sequences.

    Parameters
    ----------
    RB_number : int
        The number of sequences to construct.
    RB_length : int
        The number of Cliffords in each individual sequence.
    target : int
        Index of the target qubit

    Returns
    -------
    list
        List of RB sequences.
    """
    S = []
    for _ in range(RB_number):
        seq = np.random.choice(24, size=RB_length - 1) + 1
        seq = np.append(seq, inverseC(seq))
        seq_gates = []
        for cliff_num in seq:
            g = [f"{c}[{target}]" for c in cliffords_decomp[cliff_num - 1]]
            seq_gates.extend(g)
        S.append(seq_gates)
    return S


def inverseC(sequence):
    """Find the clifford to end a sequence s.t. it returns identity."""
    operation = Id
    for cliff in sequence:
        gate_str = "C" + str(cliff)
        gate = CLIFFORDS[gate_str]
        operation = gate @ operation
    for i in range(1, 25):
        inv = CLIFFORDS["C" + str(i)]
        trace = np.trace(inv @ operation)
        if abs(2 - abs(trace)) < 0.0001:
            return i


def perfect_cliffords(lvls: List[int], proj: str = "fulluni", num_gates: int = 1):
    """
    Legacy function to compute the clifford gates.
    """
    return CLIFFORDS


cliffords_string = [
    "C1",
    "C2",
    "C3",
    "C4",
    "C5",
    "C6",
    "C7",
    "C8",
    "C9",
    "C10",
    "C11",
    "C12",
    "C13",
    "C14",
    "C15",
    "C16",
    "C17",
    "C18",
    "C19",
    "C20",
    "C21",
    "C22",
    "C23",
    "C24",
]

cliffords_decomp = [
    ["rx90p", "rx90m"],
    ["ry90p", "rx90p"],
    ["rx90m", "ry90m"],
    ["ry90p", "rx90p", "rx90p"],
    ["rx90m"],
    ["rx90p", "ry90m", "rx90m"],
    ["rx90p", "rx90p"],
    ["ry90m", "rx90m"],
    ["rx90p", "ry90m"],
    ["ry90m"],
    ["rx90p"],
    ["rx90p", "ry90p", "rx90p"],
    ["ry90p", "ry90p"],
    ["ry90m", "rx90p"],
    ["rx90p", "ry90p"],
    ["ry90m", "rx90p", "rx90p"],
    ["rx90p", "ry90p", "ry90p"],
    ["rx90p", "ry90m", "rx90p"],
    ["rx90p", "rx90p", "ry90p", "ry90p"],
    ["ry90p", "rx90m"],
    ["rx90m", "ry90p"],
    ["ry90p"],
    ["rx90m", "ry90p", "ry90p"],
    ["rx90p", "ry90p", "rx90m"],
]

# TODO: Deal with different decompositions
# cliffords_decomp = [
#                     ['Id', 'Id', 'Id', 'Id'],
#                     ['ry90p', 'rx90p', 'Id', 'Id'],
#                     ['rx90m', 'ry90m', 'Id', 'Id'],
#                     ['ry90p', 'rx90p', 'rx90p', 'Id'],
#                     ['rx90m', 'Id', 'Id', 'Id'],
#                     ['rx90p', 'ry90m', 'rx90m', 'Id'],
#                     ['rx90p', 'rx90p', 'Id', 'Id'],
#                     ['ry90m', 'rx90m', 'Id', 'Id'],
#                     ['rx90p', 'ry90m', 'Id', 'Id'],
#                     ['ry90m', 'Id', 'Id', 'Id'],
#                     ['rx90p', 'Id', 'Id', 'Id'],
#                     ['rx90p', 'ry90p', 'rx90p', 'Id'],
#                     ['ry90p', 'ry90p', 'Id', 'Id'],
#                     ['ry90m', 'rx90p', 'Id', 'Id'],
#                     ['rx90p', 'ry90p', 'Id', 'Id'],
#                     ['ry90m', 'rx90p', 'rx90p', 'Id'],
#                     ['rx90p', 'ry90p', 'ry90p', 'Id'],
#                     ['rx90p', 'ry90m', 'rx90p', 'Id'],
#                     ['rx90p', 'rx90p', 'ry90p', 'ry90p'],
#                     ['ry90p', 'rx90m', 'Id', 'Id'],
#                     ['rx90m', 'ry90p', 'Id', 'Id'],
#                     ['ry90p', 'Id', 'Id', 'Id'],
#                     ['rx90m', 'ry90p', 'ry90p', 'Id'],
#                     ['rx90p', 'ry90p', 'rx90m', 'Id']
#                     ]


cliffords_decomp_xId = [[gate + ":Id" for gate in clif] for clif in cliffords_decomp]

sum = 0
for cd in cliffords_decomp:
    sum = sum + len(cd)
cliffors_per_gate = sum / len(cliffords_decomp)

# TODO: Deal with different decompositions
# cliffords_decomp = [
#                    ['rx90p', 'rx90m'],
#                    ['rx90p', 'rx90p'],
#                    ['ry90p', 'ry90p'],
#                    ['RZ90p', 'RZ90p'],
#                    ['rx90p'],
#                    ['ry90p'],
#                    ['RZ90p'],
#                    ['rx90m'],
#                    ['ry90m'],
#                    ['RZ90m'],
#                    ['RZ90p', 'rx90p'],
#                    ['RZ90p', 'RZ90p', 'rx90m'],
#                    ['RZ90p', 'rx90p', 'rx90p'],
#                    ['RZ90m', 'rx90p', 'rx90p'],
#                    ['RZ90p', 'rx90p'],
#                    ['RZ90p', 'rx90m'],
#                    ['rx90p', 'RZ90m'],
#                    ['RZ90p', 'RZ90p', 'ry90m'],
#                    ['RZ90p', 'ry90m'],
#                    ['RZ90m', 'ry90p'],
#                    ['RZ90p', 'RZ90p', 'ry90p'],
#                    ['RZ90m', 'rx90p'],
#                    ['RZ90p', 'ry90p'],
#                    ['RZ90m', 'rx90m']
#                ]
#
# cliffords_decomp = [
#                    ['rx90p', 'rx90m'],
#                    ['rx90p', 'rx90p'],
#                    ['ry90p', 'ry90p'],
#                    ['ry90p', 'rx90p', 'rx90p', 'ry90m'],
#                    ['rx90p'],
#                    ['ry90p'],
#                    ['ry90p', 'rx90p', 'ry90m'],
#                    ['rx90m'],
#                    ['ry90m'],
#                    ['rx90p', 'ry90p', 'rx90m'],
#                    ['ry90p', 'rx90p', 'ry90m', 'rx90p'],
#                    ['ry90p', 'rx90p', 'rx90p', 'ry90m', 'rx90m'],
#                    ['ry90p', 'rx90p', 'ry90m', 'rx90p', 'rx90p'],
#                    ['rx90p', 'ry90p', 'rx90m', 'rx90p', 'rx90p'],
#                    ['ry90p', 'rx90p', 'ry90m', 'rx90p'],
#                    ['ry90p', 'rx90p', 'ry90m', 'rx90m'],
#                    ['rx90p', 'rx90p', 'ry90p', 'rx90m'],
#                    ['ry90p', 'rx90p', 'rx90p', 'ry90m', 'ry90m'],
#                    ['ry90p', 'rx90p', 'ry90m', 'ry90m'],
#                    ['rx90p', 'ry90p', 'rx90m', 'ry90p'],
#                    ['ry90p', 'rx90p', 'rx90p'],
#                    ['rx90p', 'ry90p'],
#                    ['ry90p', 'rx90p'],
#                    ['rx90p', 'ry90p', 'rx90m', 'rx90m']
#                ]
#
# cliffords_decomp = [
#                    ['rx90p', 'rx90m'],
#                    ['ry90p', 'rx90p'],
#                    ['rx90m', 'ry90m'],
#                    ['ry90p', 'rxp'],
#                    ['rx90m'],
#                    ['rx90p', 'ry90m', 'rx90m'],
#                    ['rxp'],
#                    ['ry90m', 'rx90m'],
#                    ['rx90p', 'ry90m'],
#                    ['ry90m'],
#                    ['rx90p'],
#                    ['rx90p', 'ry90p', 'rx90p'],
#                    ['ryp'],
#                    ['ry90m', 'rx90p'],
#                    ['rx90p', 'ry90p'],
#                    ['ry90m', 'rxp'],
#                    ['rx90p', 'ryp'],
#                    ['rx90p', 'ry90m', 'rx90p'],
#                    ['rxp', 'ryp'],
#                    ['ry90p', 'rx90m'],
#                    ['rx90m', 'ry90p'],
#                    ['ry90p'],
#                    ['rx90m', 'ryp'],
#                    ['rx90p', 'ry90p', 'rx90m']
#                    ]
