"""Useful functions to get basis vectors and matrices of the right size."""

import numpy as np
from typing import List
from scipy.linalg import block_diag as scipy_block_diag
from scipy.linalg import expm
from c3.libraries.constants import Id, X, Y, Z, PAULIS, GATES


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
    nq = len(dims)
    _SINGLE_QUBIT_PAULI_BASIS = (Id, X, Y, Z)
    paulis = []
    for dim in dims:
        paulis.append([P for P in _SINGLE_QUBIT_PAULI_BASIS])
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
        op = op_tuple[0]
        for i in range(nq - 1):
            op = np.kron(op, op_tuple[i + 1])
        vec = np.reshape(np.transpose(op), [-1, 1])
        B[:, idx] = vec.T.conj()
    return B


# MATH HELPERS
def np_kron_n(mat_list):
    """
    Apply Kronecker product to a list of matrices.
    """
    tmp = mat_list[0]
    for m in mat_list[1:]:
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


def projector(dims, indices):
    """
    Computes the projector to cut down a matrix to the computational space. The
    subspaces indicated in indeces will be projected to the lowest two states,
    the rest is projected onto the lowest state.
    """
    ids = []
    for index, dim in enumerate(dims):
        if index in indices:
            ids.append(np.eye(dim, 2))
        else:
            ids.append(np.eye(dim, 1))
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


# def perfect_gate(  # noqa
#     gate_str: str, index=[0, 1], dims=[2, 2], proj: str = "wzeros"
# ):
#     """
#     Construct an ideal single or two-qubit gate.
#
#     Parameters
#     ----------
#     gate_str: str
#         Identifier of the gate, i.e. "rx90p".
#     index : list
#         Indeces of the subspace(s) the gate acts on
#     dims : list
#         Dimension of the subspace(s)
#     proj : str
#         Option for projection in the case of more than two-level qubits.
#
#     Returns
#     -------
#     np.array
#         Ideal representation of the gate.
#
#     """
#     do_pad_gate = True
#     kron_list = []
#     for dim in dims:
#         kron_list.append(np.eye(dim))
#     lvls = dims[index[0]]
#     if gate_str in GATES.keys():
#         gate = GATES[gate_str]
#     elif gate_str == "CNOT":
#         raise NotImplementedError(
#             "A correct implementation of perfect CNOT is pending, use Crxp now"
#         )
#     elif gate_str == "CR":
#         # TODO: Fix the ideal CNOT construction.
#         lvls2 = dims[gate_num + 1]
#         Z = 1j * perfect_gate("RZp", index, [lvls], proj)
#         X = perfect_gate("rxp", index, [lvls2], proj)
#         gate = np.kron(Z, X)
#         gate_num += 1
#         do_pad_gate = False
#     elif gate_str == "CR90":
#         lvls2 = dims[gate_num + 1]
#         rxp_temp = perfect_gate("rx90p", index, [lvls2], proj)
#         rxm_temp = perfect_gate("rx90m", index, [lvls2], proj)
#         gate = scipy_block_diag(rxp_temp, rxm_temp)
#         for ii in range(2, lvls):
#             gate = pad_matrix(gate, lvls2, proj)
#         gate_num += 1
#         do_pad_gate = False
#     elif gate_str == "iSWAP":
#         # TODO make construction of iSWAP work with superoperator too
#         lvls2 = dims[gate_num + 1]
#         if lvls == 2 and lvls2 == 2:
#             gate = ISWAP
#         elif lvls == 3 and lvls2 == 3:
#             gate = ISWAP3
#         gate_num += 1
#         do_pad_gate = False
#     else:
#         raise KeyError(
#             "gate_str must be one of the basic 90 or 180 degree gates: 'Id','rx90p','rx90m','rxp','ry90p','ry90m','ryp','RZ90p','RZ90m','RZp', 'CNOT', Crxp, CRZp, CR, CR90, iSWAP"
#         )
#     if do_pad_gate:
#         if proj == "compsub":
#             pass
#         elif proj == "wzeros":
#             zeros = np.zeros([lvls - 2, lvls - 2])
#             gate = scipy_block_diag(gate, zeros)
#         elif proj == "fulluni":
#             identity = np.eye(lvls - 2)
#             gate = scipy_block_diag(gate, identity)
#         else:
#             raise KeyError("proj should be wzeros, compsub or fulluni")
#     kron_list[index[0]] = gate
#     return np_kron_n(kron_list)


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
    target : str
        Which qubit is measured. Options: "left" or "right"

    Returns
    -------
    list
        Relaxation sequence.

    """
    wait = ["Id:Id"]
    if target == "left":
        prepare_1 = ["rx90p:Id", "rx90p:Id"]
    elif target == "right":
        prepare_1 = ["Id:rx90p", "Id:rx90p"]
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
    wait = ["Id:Id"]
    if target == "left":
        rotate_90 = ["rx90p:Id"]
    elif target == "right":
        rotate_90 = ["Id:rx90p"]
    S = []
    S.extend(rotate_90)
    S.extend(wait * length)
    S.extend(rotate_90)
    return S


def ramsey_echo_sequence(length, target):
    """
    Generate a gate sequence to measure dephasing time in a two-qubit chip including a flip in the middle.
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
    wait = ["Id:Id"]
    hlength = length // 2
    if target == "left":
        rotate_90_p = ["rx90p:Id"]
        rotate_90_m = ["rx90m:Id"]
    elif target == "right":
        rotate_90_p = ["Id:rx90p"]
        rotate_90_m = ["Id:rx90m"]
    S = []
    S.extend(rotate_90_p)
    S.extend(wait * hlength)
    S.extend(rotate_90_p)
    S.extend(rotate_90_p)
    S.extend(wait * hlength)
    S.extend(rotate_90_m)
    return S


def single_length_RB(RB_number, RB_length, padding=""):
    """Given a length and number of repetitions it compiles Randomized Benchmarking sequences.

    Parameters
    ----------
    RB_number : int
        The number of sequences to construct.
    RB_length : int
        The number of Cliffords in each individual sequence.
    padding : str
        Option of "left" or "right" in a two-qubit chip.

    Returns
    -------
    list
        List of RB sequences.
    """
    S = []
    for seq_idx in range(RB_number):
        seq = np.random.choice(24, size=RB_length - 1) + 1
        seq = np.append(seq, inverseC(seq))
        seq_gates = []
        for cliff_num in seq:
            # TODO: General padding for n qubits
            if padding == "right":
                g = ["Id:" + c for c in cliffords_decomp[cliff_num - 1]]
            elif padding == "left":
                g = [c + ":Id" for c in cliffords_decomp[cliff_num - 1]]
            else:
                g = cliffords_decomp[cliff_num - 1]
            seq_gates.extend(g)
        S.append(seq_gates)
    return S


def inverseC(sequence):
    """Find the clifford to end a sequence s.t. it returns identity."""
    operation = Id
    for cliff in sequence:
        gate_str = "C" + str(cliff)
        gate = eval(gate_str)
        operation = gate @ operation
    for i in range(1, 25):
        inv = eval("C" + str(i))
        trace = np.trace(inv @ operation)
        if abs(2 - abs(trace)) < 0.0001:
            return i


def perfect_cliffords(lvls: List[int], proj: str = "fulluni", num_gates: int = 1):
    """
    Returns a list of ideal matrix representation of Clifford gates.
    """
    # TODO make perfect clifford more general by making it take a decomposition

    if num_gates == 1:
        x90p = GATES["rx90p"]
        y90p = GATES["ry90p"]
        x90m = GATES["rx90m"]
        y90m = GATES["ry90m"]
    elif num_gates == 2:
        x90p = GATES["rx90p"]
        y90p = GATES["ry90p"]
        x90m = GATES["rx90m"]
        y90m = GATES["ry90m"]

    C1 = x90m @ x90p
    C2 = x90p @ y90p
    C3 = y90m @ x90m
    C4 = x90p @ x90p @ y90p
    C5 = x90m
    C6 = x90m @ y90m @ x90p
    C7 = x90p @ x90p
    C8 = x90m @ y90m
    C9 = y90m @ x90p
    C10 = y90m
    C11 = x90p
    C12 = x90p @ y90p @ x90p
    C13 = y90p @ y90p
    C14 = x90p @ y90m
    C15 = y90p @ x90p
    C16 = x90p @ x90p @ y90m
    C17 = y90p @ y90p @ x90p
    C18 = x90p @ y90m @ x90p
    C19 = y90p @ y90p @ x90p @ x90p
    C20 = x90m @ y90p
    C21 = y90p @ x90m
    C22 = y90p
    C23 = y90p @ y90p @ x90m
    C24 = x90m @ y90p @ x90p

    cliffords = [
        C1,
        C2,
        C3,
        C4,
        C5,
        C6,
        C7,
        C8,
        C9,
        C10,
        C11,
        C12,
        C13,
        C14,
        C15,
        C16,
        C17,
        C18,
        C19,
        C20,
        C21,
        C22,
        C23,
        C24,
    ]

    return cliffords


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

cliffords_decomp_xId = [[gate + ":Id" for gate in clif] for clif in cliffords_decomp]

sum = 0
for cd in cliffords_decomp:
    sum = sum + len(cd)
cliffors_per_gate = sum / len(cliffords_decomp)

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
