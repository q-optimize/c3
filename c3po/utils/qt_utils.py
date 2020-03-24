"""Useful fuctions to get basis vectors and matrices of the right size."""

import numpy as np
from scipy.linalg import block_diag as scipy_block_diag
from scipy.linalg import expm

# Pauli matrices
Id = np.array([[1, 0],
               [0, 1]],
              dtype=np.complex128)
X = np.array([[0, 1],
              [1, 0]],
             dtype=np.complex128)
Y = np.array([[0, -1j],
              [1j, 0]],
             dtype=np.complex128)
Z = np.array([[1, 0],
              [0, -1]],
             dtype=np.complex128)

# TODO Combine the above Pauli definitions with this dict. Move to constants.
PAULIS = {
    "X": X,
    "Y": Y,
    "Z": Z,
    "Id": Id
}

# MATH HELPERS
def np_kron_n(mat_list):
    tmp = mat_list[0]
    for m in mat_list[1:]:
        tmp = np.kron(tmp, m)
    return tmp


def hilbert_space_kron(op, indx, dims):
    """
    Extend an operator op on subspace indx to the full product hilbert space
    given by dimensions in dims.
    """
    op_list = []
    for indy in range(len(dims)):
        qI = np.identity(dims[indy])
        if indy == indx:
            op_list.append(op)
        else:
            op_list.append(qI)
    if indx > len(dims) - 1:
        raise Warning(
            f"Index {indx} is outside the Hilbert space dimensions {dims}. "
        )
    return(np_kron_n(op_list))


def hilbert_space_dekron(op, indx, dims):
    """
    Partial trace of an operator to return equivalent subspace operator.
    Inverse of hilbert_space_kron.
    """
    # TODO Partial trace, reducing operators and states to subspace.
    pass


def rotation(phase, xyz):
    """General Rotation."""
    rot = np.cos(phase/2) * Id - \
        1j * np.sin(phase/2) * (xyz[0] * X + xyz[1] * Y + xyz[2] * Z)
    return rot


X90p = rotation(np.pi/2, [1, 0, 0])  # Rx+
X90m = rotation(-np.pi/2, [1, 0, 0])  # Rx-
Xp = rotation(np.pi, [1, 0, 0])
Y90p = rotation(np.pi/2, [0, 1, 0])  # Ry+
Y90m = rotation(-np.pi/2, [0, 1, 0])  # Ry-
Yp = rotation(np.pi, [0, 1, 0])
Z90p = rotation(np.pi/2, [0, 0, 1])  # Rz+
Z90m = rotation(-np.pi/2, [0, 0, 1])  # Rz-
Zp = rotation(np.pi, [0, 0, 1])


def basis(lvls: int, pop_lvl: int):
    psi = np.zeros([lvls, 1])
    psi[pop_lvl] = 1
    return psi


def xy_basis(lvls: int, vect: str):
    psi_g = basis(lvls, 0)
    psi_e = basis(lvls, 1)
    if vect == 'zm':
        psi = psi_g
    elif vect == 'zp':
        psi = psi_e
    elif vect == 'xp':
        psi = (psi_g + psi_e) / np.sqrt(2)
    elif vect == 'xm':
        psi = (psi_g - psi_e) / np.sqrt(2)
    elif vect == 'yp':
        psi = (psi_g + 1.0j * psi_e) / np.sqrt(2)
    elif vect == 'ym':
        psi = (psi_g - 1.0j * psi_e) / np.sqrt(2)
    else:
        print("vect must be one of \'zp\' \'zm\' \'xp\' \'xm\' \'yp\' \'ym\'")
        return None
    return psi


def pad_matrix(matrix, dim, padding):
    """
    Fills matrix dimsions with zeros or identity.
    """
    if padding == 'compsub':
        return matrix
    elif padding == 'wzeros':
        zeros = np.zeros([dim, dim])
        matrix = scipy_block_diag(matrix, zeros)
    elif padding == 'fulluni':
        identity = np.eye(dim)
        matrix = scipy_block_diag(matrix, identity)
    return matrix


def perfect_gate(
    gates_str: str, index=[0, 1], dims=[2, 2], proj: str = 'wzeros'
):
    do_pad_gate = True
    # TODO index for now unused
    kron_list = []
    # for dim in dims:
    #     kron_list.append(np.eye(dim))
    kron_gate = 1
    gate_num = 0
    # Note that the gates_str has to be explicit for all subspaces
    # (and ordered)
    for gate_str in gates_str.split(":"):
        lvls = dims[gate_num]
        if gate_str == 'Id':
            gate = Id
        elif gate_str == 'X90p':
            gate = X90p
        elif gate_str == 'X90m':
            gate = X90m
        elif gate_str == 'Xp':
            gate = Xp
        elif gate_str == 'Y90p':
            gate = Y90p
        elif gate_str == 'Y90m':
            gate = Y90m
        elif gate_str == 'Yp':
            gate = Yp
        elif gate_str == 'Z90p':
            gate = Z90p
        elif gate_str == 'Z90m':
            gate = Z90m
        elif gate_str == 'Zp':
            gate = Zp
        elif gate_str == 'CNOT':
            # TODO: Fix the ideal CNOT construction.
            lvls2 = dims[gate_num + 1]
            NOT = 1j*perfect_gate('Xp', index, [lvls2], proj)
            C = perfect_gate('Id', index, [lvls2], proj)
            gate = scipy_block_diag(C, NOT)
            # We increase gate_num since CNOT is a two qubit gate
            for ii in range(2, lvls):
                pad_matrix(gate, lvls2, proj)
            gate_num += 1
            do_pad_gate = False
        elif gate_str == 'CZ':
            # TODO: Fix the ideal CNOT construction.
            lvls2 = dims[gate_num + 1]
            NOT = 1j*perfect_gate('Zp', index, [lvls2], proj)
            C = perfect_gate('Id', index, [lvls2], proj)
            gate = scipy_block_diag(C, NOT)
            # We increase gate_num since CNOT is a two qubit gate
            for ii in range(2, lvls):
                pad_matrix(gate, lvls2, proj)
            gate_num += 1
            do_pad_gate = False
        elif gate_str == 'CR':
            # TODO: Fix the ideal CNOT construction.
            lvls2 = dims[gate_num + 1]
            Z = 1j*perfect_gate('Zp', index, [lvls], proj)
            X = perfect_gate('Xp', index, [lvls2], proj)
            gate = np.kron(Z, X)
            gate_num += 1
            do_pad_gate = False
        elif gate_str == 'CR90':
            # TODO: Fix the ideal CNOT construction.
            lvls2 = dims[gate_num + 1]
            Z = 1j*perfect_gate('Z90p', index, [lvls], proj)
            X = perfect_gate('X90p', index, [lvls2], proj)
            gate = np.kron(Z, X)
            gate_num += 1
            do_pad_gate = False
        else:
            print("gate_str must be one of the basic 90 or 180 degree gates.")
            print("\'Id\',\'X90p\',\'X90m\',\'Xp\',\'Y90p\',",
                  "\'Y90m\',\'Yp\',\'Z90p\',\'Z90m\',\'Zp\', \'CNOT\'")
            return None
        if do_pad_gate:
            if proj == 'compsub':
                pass
            elif proj == 'wzeros':
                zeros = np.zeros([lvls - 2, lvls - 2])
                gate = scipy_block_diag(gate, zeros)
            elif proj == 'fulluni':
                identity = np.eye(lvls - 2)
                gate = scipy_block_diag(gate, identity)
        kron_list.append(gate)
        gate_num += 1
    return np_kron_n(kron_list)

def perfect_parametric_gate(paulis_str, ang, dims):
    ps = []
    p_list = paulis_str.split(":")
    for idx in range(len(p_list)):
        if p_list[idx] not in PAULIS:
            raise KeyError(
                f"Incorrect pauli matrix {p_list[idx]} in position {idx}.\
                Select from {PAULIS.keys()}."
            )
        ps.append(
                pad_matrix(PAULIS[p_list[idx]], dims[idx]-2, "wzeros")
        )
    gen = np_kron_n(ps)
    return expm(-1.j/2 * ang * gen)


def perfect_CZ(lvls: int, proj: str = 'wzeros'):
    Id = perfect_gate(lvls, 'Id', proj=proj)
    CZ = np.kron(Id,Id)
    if proj == 'compsub':
        CZ[3,3] = -1
    elif proj == 'wzeros' or proj == 'fulluni':
        CZ[lvls+1,lvls+1] = -1
    return CZ


def single_length_RB(RB_number, RB_length, padding=""):
    """Given a length and number of repetitions it compiles RB sequences."""
    S = []
    for seq_idx in range(RB_number):
        seq = np.random.choice(24, size=RB_length-1)+1
        seq = np.append(seq, inverseC(seq))
        seq_gates = []
        for cliff_num in seq:
            # TODO: General padding for n qubits
            if padding == "right":
                g = ["Id:" + c for c in cliffords_decomp[cliff_num-1]]
            elif padding == "left":
                g = [c + ":Id" for c in cliffords_decomp[cliff_num-1]]
            else:
                g = cliffords_decomp[cliff_num-1]
            seq_gates.extend(g)
        S.append(seq_gates)
    return S


def inverseC(sequence):
    """Find the clifford to end a sequence s.t. it returns identity."""
    operation = Id
    for cliff in sequence:
        gate_str = 'C'+str(cliff)
        gate = eval(gate_str)
        operation = gate @ operation
    for i in range(1, 25):
        inv = eval('C'+str(i))
        trace = np.trace(inv @ operation)
        if abs(2 - abs(trace)) < 0.0001:
            return i


C1 = X90m @ X90p
C2 = X90p @ Y90p
C3 = Y90m @ X90m
C4 = X90p @ X90p @ Y90p
C5 = X90m
C6 = X90m @ Y90m @ X90p
C7 = X90p @ X90p
C8 = X90m @ Y90m
C9 = Y90m @ X90p
C10 = Y90m
C11 = X90p
C12 = X90p @ Y90p @ X90p
C13 = Y90p @ Y90p
C14 = X90p @ Y90m
C15 = Y90p @ X90p
C16 = X90p @ X90p @ Y90m
C17 = Y90p @ Y90p @ X90p
C18 = X90p @ Y90m @ X90p
C19 = Y90p @ Y90p @ X90p @ X90p
C20 = X90m @ Y90p
C21 = Y90p @ X90m
C22 = Y90p
C23 = Y90p @ Y90p @ X90m
C24 = X90m @ Y90p @ X90p


def perfect_cliffords(lvls: int, proj: str = 'fulluni', num_gates: int = 1):
    # TODO make perfect clifford more general by making it take a decomposition

    if num_gates == 1:
        x90p = perfect_gate(lvls, 'X90p', proj)
        y90p = perfect_gate(lvls, 'Y90p', proj)
        x90m = perfect_gate(lvls, 'X90m', proj)
        y90m = perfect_gate(lvls, 'Y90m', proj)
    elif num_gates == 2:
        x90p = perfect_gate(lvls, 'X90p:Id', proj)
        y90p = perfect_gate(lvls, 'Y90p:Id', proj)
        x90m = perfect_gate(lvls, 'X90m:Id', proj)
        y90m = perfect_gate(lvls, 'Y90m:Id', proj)

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

    cliffords = [C1, C2, C3, C4, C5, C6, C7, C8, C9, C10, C11, C12, C13,
                 C14, C15, C16, C17, C18, C19, C20, C21, C22, C23, C24]

    return cliffords

cliffords_string = ['C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7', 'C8', 'C9',
                    'C10', 'C11', 'C12', 'C13', 'C14', 'C15', 'C16', 'C17',
                    'C18', 'C19', 'C20', 'C21', 'C22', 'C23', 'C24']

cliffords_decomp = [
                    ['X90p', 'X90m'],
                    ['Y90p', 'X90p'],
                    ['X90m', 'Y90m'],
                    ['Y90p', 'X90p', 'X90p'],
                    ['X90m'],
                    ['X90p', 'Y90m', 'X90m'],
                    ['X90p', 'X90p'],
                    ['Y90m', 'X90m'],
                    ['X90p', 'Y90m'],
                    ['Y90m'],
                    ['X90p'],
                    ['X90p', 'Y90p', 'X90p'],
                    ['Y90p', 'Y90p'],
                    ['Y90m', 'X90p'],
                    ['X90p', 'Y90p'],
                    ['Y90m', 'X90p', 'X90p'],
                    ['X90p', 'Y90p', 'Y90p'],
                    ['X90p', 'Y90m', 'X90p'],
                    ['X90p', 'X90p', 'Y90p', 'Y90p'],
                    ['Y90p', 'X90m'],
                    ['X90m', 'Y90p'],
                    ['Y90p'],
                    ['X90m', 'Y90p', 'Y90p'],
                    ['X90p', 'Y90p', 'X90m']
                    ]

cliffords_decomp_xId = [
    [gate + ':Id' for gate in clif] for clif in cliffords_decomp
]

sum = 0
for cd in cliffords_decomp:
    sum = sum + len(cd)
cliffors_per_gate = sum / len(cliffords_decomp)

# cliffords_decomp = [
#                    ['X90p', 'X90m'],
#                    ['X90p', 'X90p'],
#                    ['Y90p', 'Y90p'],
#                    ['Z90p', 'Z90p'],
#                    ['X90p'],
#                    ['Y90p'],
#                    ['Z90p'],
#                    ['X90m'],
#                    ['Y90m'],
#                    ['Z90m'],
#                    ['Z90p', 'X90p'],
#                    ['Z90p', 'Z90p', 'X90m'],
#                    ['Z90p', 'X90p', 'X90p'],
#                    ['Z90m', 'X90p', 'X90p'],
#                    ['Z90p', 'X90p'],
#                    ['Z90p', 'X90m'],
#                    ['X90p', 'Z90m'],
#                    ['Z90p', 'Z90p', 'Y90m'],
#                    ['Z90p', 'Y90m'],
#                    ['Z90m', 'Y90p'],
#                    ['Z90p', 'Z90p', 'Y90p'],
#                    ['Z90m', 'X90p'],
#                    ['Z90p', 'Y90p'],
#                    ['Z90m', 'X90m']
#                ]
#
# cliffords_decomp = [
#                    ['X90p', 'X90m'],
#                    ['X90p', 'X90p'],
#                    ['Y90p', 'Y90p'],
#                    ['Y90p', 'X90p', 'X90p', 'Y90m'],
#                    ['X90p'],
#                    ['Y90p'],
#                    ['Y90p', 'X90p', 'Y90m'],
#                    ['X90m'],
#                    ['Y90m'],
#                    ['X90p', 'Y90p', 'X90m'],
#                    ['Y90p', 'X90p', 'Y90m', 'X90p'],
#                    ['Y90p', 'X90p', 'X90p', 'Y90m', 'X90m'],
#                    ['Y90p', 'X90p', 'Y90m', 'X90p', 'X90p'],
#                    ['X90p', 'Y90p', 'X90m', 'X90p', 'X90p'],
#                    ['Y90p', 'X90p', 'Y90m', 'X90p'],
#                    ['Y90p', 'X90p', 'Y90m', 'X90m'],
#                    ['X90p', 'X90p', 'Y90p', 'X90m'],
#                    ['Y90p', 'X90p', 'X90p', 'Y90m', 'Y90m'],
#                    ['Y90p', 'X90p', 'Y90m', 'Y90m'],
#                    ['X90p', 'Y90p', 'X90m', 'Y90p'],
#                    ['Y90p', 'X90p', 'X90p'],
#                    ['X90p', 'Y90p'],
#                    ['Y90p', 'X90p'],
#                    ['X90p', 'Y90p', 'X90m', 'X90m']
#                ]
#
# cliffords_decomp = [
#                    ['X90p', 'X90m'],
#                    ['Y90p', 'X90p'],
#                    ['X90m', 'Y90m'],
#                    ['Y90p', 'Xp'],
#                    ['X90m'],
#                    ['X90p', 'Y90m', 'X90m'],
#                    ['Xp'],
#                    ['Y90m', 'X90m'],
#                    ['X90p', 'Y90m'],
#                    ['Y90m'],
#                    ['X90p'],
#                    ['X90p', 'Y90p', 'X90p'],
#                    ['Yp'],
#                    ['Y90m', 'X90p'],
#                    ['X90p', 'Y90p'],
#                    ['Y90m', 'Xp'],
#                    ['X90p', 'Yp'],
#                    ['X90p', 'Y90m', 'X90p'],
#                    ['Xp', 'Yp'],
#                    ['Y90p', 'X90m'],
#                    ['X90m', 'Y90p'],
#                    ['Y90p'],
#                    ['X90m', 'Yp'],
#                    ['X90p', 'Y90p', 'X90m']
#                    ]
