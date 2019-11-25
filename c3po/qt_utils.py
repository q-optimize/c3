"""Useful fuctions to get basis vectors and matrices of the right size."""

import numpy as np
from scipy.linalg import block_diag as scipy_block_diag

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

def perfect_gate(lvls: int, gate_str: str, proj: bool = True):
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
    else:
        print("gate_str must be one of the basic 90 or 180 degree gates.")
        print("\'Id\',\'X90p\',\'X90m\',\'Xp\',\'Y90p\',",
              "\'Y90m\',\'Yp\',\'Z90p\',\'Z90m\',\'Zp\'")
        return None
    if proj:
        zeros = np.zeros([lvls - 2, lvls - 2])
        gate = scipy_block_diag(gate, zeros)
    else:
        identity = np.eye(lvls - 2)
        gate = scipy_block_diag(gate, identity)
    return gate


def single_length_RB(RB_number, RB_length):
    """Given a length and number of repetitions it compiles RB sequences."""
    S = []
    for seq_idx in range(RB_number):
        seq = np.random.choice(24, size=RB_length-1)+1
        seq = np.append(seq, inverseC(seq))
        seq_gates = []
        for cliff_num in seq:
            seq_gates.extend(cliffords_decomp[cliff_num-1])
        S.append(seq_gates)
    return S


def inverseC(sequence):
    """Find the clifford to end a sequence s.t. it returns identity."""
    operation = Id
    for cliff in sequence:
        gate_str = 'C'+str(cliff)
        gate = eval(gate_str)
        operation = operation @ gate
    for i in range(1, 25):
        inv = eval('C'+str(i))
        trace = np.trace(operation @ inv)
        if abs(2 - abs(trace)) < 0.0001:
            return i

# C1 = I
# C2 = X90p @ X90p
# C3 = Y90p @ Y90p
# C4 = Z90p @ Z90p
# C5 = X90p
# C6 = Y90p
# C7 = Z90p
# C8 = X90m
# C9 = Y90m
# C10 = Z90m
# C11 = Z90p @ X90p
# C12 = Z90p @ Z90p @ X90m
# C13 = Z90p @ X
# C14 = Z90m @ X
# C15 = Z90p @ X90p
# C16 = Z90p @ X90m
# C17 = X90p @ Z90m
# C18 = Z90p @ Z90p @ Y90m
# C19 = Z90p @ Y90m
# C20 = Z90m @ Y90p
# C21 = Z90p @ Z90p @ Y90p #Hadamard
# C22 = Z90m @ X90p
# C23 = Z90p @ Y90p
# C24 = Z90m @ X90m

def perfect_cliffords(lvls: int):
    # TODO make perfect clifford more general by making it take a decomposition
    identity = np.eye(lvls - 2)
    x90p = scipy_block_diag(X90p, identity)
    y90p = scipy_block_diag(Y90p, identity)
    x90m = scipy_block_diag(X90m, identity)
    y90m = scipy_block_diag(Y90m, identity)

    C1 = x90p @ x90m
    C2 = y90p @ x90p
    C3 = x90m @ y90m
    C4 = y90p @ x90p @ x90p
    C5 = x90m
    C6 = x90p @ y90m @ x90m
    C7 = x90p @ x90p
    C8 = y90m @ x90m
    C9 = x90p @ y90m
    C10 = y90m
    C11 = x90p
    C12 = x90p @ y90p @ x90p
    C13 = y90p @ y90p
    C14 = y90m @ x90p
    C15 = x90p @ y90p
    C16 = y90m @ x90p @ x90p
    C17 = x90p @ y90p @ y90p
    C18 = x90p @ y90m @ x90p
    C19 = x90p @ x90p @ y90p @ y90p
    C20 = y90p @ x90m
    C21 = x90m @ y90p
    C22 = y90p
    C23 = x90m @ y90p @ y90p
    C24 = x90p @ y90p @ x90m

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
