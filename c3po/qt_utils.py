import numpy as np
import tensorflow as tf
from scipy.linalg import block_diag

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
    rot = np.cos(phase/2) * Id
    - 1j * np.sin(phase/2) * (xyz[0] * X + xyz[1] * Y + xyz[2] * Z)
    return rot


X90p = rotation(np.pi/2, [1, 0, 0])  # Rx+
X90m = rotation(-np.pi/2, [1, 0, 0])  # Rx-
Xp = rotation(np.pi, [1, 0, 0])
Y90p = rotation(np.pi/2, [0, 1, 0])  # Ry+
Y90m = rotation(-np.pi/2, [0, 1, 0])  # Ry-
Yp = rotation(np.pi, [0, 1, 0])
Z90p = rotation(np.pi/2, [0, 0, 1])  # Rz+
Z90m = rotation(-np.pi/2, [0, 0, 1])  # Rz-

# Define states & unitaries
# TODO create basic unit vectors with a function call or an import
psi_g = np.zeros([qubit_lvls, 1])
psi_g[0] = 1
psi_e = np.zeros([qubit_lvls, 1])
psi_e[1] = 1
ket_0 = tf.constant(psi_g, dtype=tf.complex128)
bra_1 = tf.constant(psi_e.T, dtype=tf.complex128)
psi_ym = (psi_g - 1.0j * psi_e) / np.sqrt(2)
bra_ym = tf.constant(psi_ym.T, dtype=tf.complex128)
psi_yp = (psi_g + 1.0j * psi_e) / np.sqrt(2)
bra_yp = tf.constant(psi_yp.T, dtype=tf.complex128)
psi_xm = (psi_g - psi_e) / np.sqrt(2)
bra_xm = tf.constant(psi_xm.T, dtype=tf.complex128)
psi_xp = (psi_g + psi_e) / np.sqrt(2)
bra_xp = tf.constant(psi_xp.T, dtype=tf.complex128)

diag = np.zeros([qubit_lvls, 1])
diag[0:2] = 1
idm = tf.constant(np.diag(diag), dtype=tf.complex128)


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
    operation = I
    for cliff in sequence:
        gate_str = 'C'+str(cliff)
        gate = eval(gate_str)
        operation = operation @ gate
    for i in range(1, 25):
        inv = eval('C'+str(i))
        trace = np.trace(operation @ inv)
        if abs(2 - abs(trace)) < 0.0001:
            return i


############ GATES ############





# #TODO replace rotation with expm
# def rotation(phase,x,y,z):
#     return (-1.0j*(phase/2)*Qobj(x*X+y*Y+z*Z)).expm()


basic_gates_dict = {
        'X90p':Qobj(X90p),
        'X90m':Qobj(X90m),
        'Y90p':Qobj(Y90p),
        'Y90m':Qobj(Y90m),
        'Z90p':Qobj(Z90p),
        'Z90m':Qobj(Z90m)
        }

def overrot_basic_gates(over_rotation_percentage = 0.05):
    rot = 1 + over_rotation_percentage
    X90p_or=rotation(rot*np.pi/2,1,0,0) #Rx+
    X90m_or=rotation(rot*-np.pi/2,1,0,0) #Rx-
    Y90p_or=rotation(rot*np.pi/2,0,1,0) #Ry+
    Y90m_or=rotation(rot*-np.pi/2,0,1,0) #Ry-
    Z90p_or=rotation(rot*np.pi/2,0,0,1) #Rz+
    Z90m_or=rotation(rot*-np.pi/2,0,0,1) #Rz-
    overrot_basic_gates_dict = {
            'X90p':Qobj(X90p_or),
            'X90m':Qobj(X90m_or),
            'Y90p':Qobj(Y90p_or),
            'Y90m':Qobj(Y90m_or),
            'Z90p':Qobj(Z90p_or),
            'Z90m':Qobj(Z90m_or)
            }
    return overrot_basic_gates_dict

#C1=I
#C2=X90p@X90p
#C3=Y90p@Y90p
#C4=Z90p@Z90p
#C5=X90p
#C6=Y90p
#C7=Z90p
#C8=X90m
#C9=Y90m
#C10=Z90m
#C11=Z90p@X90p
#C12=Z90p@Z90p@X90m
#C13=Z90p@X
#C14=Z90m@X
#C15=Z90p@X90p
#C16=Z90p@X90m
#C17=X90p@Z90m
#C18=Z90p@Z90p@Y90m
#C19=Z90p@Y90m
#C20=Z90m@Y90p
#C21=Z90p@Z90p@Y90p #Hadamard
#C22=Z90m@X90p
#C23=Z90p@Y90p
#C24=Z90m@X90m

C1 = I
C2 = Y90p @ X90p
C3 = X90m @ Y90m
C4 = Y90p @ Xp
C5 = X90m
C6 = X90p @ Y90m @ X90m
C7 = Xp
C8 = Y90m @ X90m
C9 = X90p @ Y90m
C10 = Y90m
C11 = X90p
C12 = X90p @ Y90p @ X90p
C13 = Yp
C14 = Y90m @ X90p
C15 = X90p @ Y90p
C16 = Y90m @ Xp
C17 = X90p @ Yp
C18 = X90p @ Y90m @ X90p
C19 = Xp @ Yp
C20 = Y90p @ X90m
C21 = X90m @ Y90p
C22 = Y90p
C23 = X90m @ Yp
C24 = X90p @ Y90p @ X90m

cliffords=[C1,C2,C3,C4,C5,C6,C7,C8,C9,C10,C11,C12,C13,
           C14,C15,C16,C17,C18,C19,C20,C21,C22,C23,C24]

cliffords_string=['C1','C2','C3','C4','C5','C6','C7','C8','C9','C10','C11','C12','C13',
                  'C14','C15','C16','C17','C18','C19','C20','C21','C22','C23','C24']
#cliffords_decomp=[
#                    ['X90p','X90m'],
#                    ['X90p','X90p'],
#                    ['Y90p','Y90p'],
#                    ['Z90p','Z90p'],
#                    ['X90p'],
#                    ['Y90p'],
#                    ['Z90p'],
#                    ['X90m'],
#                    ['Y90m'],
#                    ['Z90m'],
#                    ['Z90p','X90p'],
#                    ['Z90p','Z90p','X90m'],
#                    ['Z90p','X90p','X90p'],
#                    ['Z90m','X90p','X90p'],
#                    ['Z90p','X90p'],
#                    ['Z90p','X90m'],
#                    ['X90p','Z90m'],
#                    ['Z90p','Z90p','Y90m'],
#                    ['Z90p','Y90m'],
#                    ['Z90m','Y90p'],
#                    ['Z90p','Z90p','Y90p'],
#                    ['Z90m','X90p'],
#                    ['Z90p','Y90p'],
#                    ['Z90m','X90m']
#                ]

#cliffords_decomp=[
#                    ['X90p','X90m'],
#                    ['X90p','X90p'],
#                    ['Y90p','Y90p'],
#                    ['Y90p','X90p','X90p','Y90m'],
#                    ['X90p'],
#                    ['Y90p'],
#                    ['Y90p','X90p','Y90m'],
#                    ['X90m'],
#                    ['Y90m'],
#                    ['X90p','Y90p','X90m'],
#                    ['Y90p','X90p','Y90m','X90p'],
#                    ['Y90p','X90p','X90p','Y90m','X90m'],
#                    ['Y90p','X90p','Y90m','X90p','X90p'],
#                    ['X90p','Y90p','X90m','X90p','X90p'],
#                    ['Y90p','X90p','Y90m','X90p'],
#                    ['Y90p','X90p','Y90m','X90m'],
#                    ['X90p','X90p','Y90p','X90m'],
#                    ['Y90p','X90p','X90p','Y90m','Y90m'],
#                    ['Y90p','X90p','Y90m','Y90m'],
#                    ['X90p','Y90p','X90m','Y90p'],
#                    ['Y90p','X90p','X90p'],
#                    ['X90p','Y90p'],
#                    ['Y90p','X90p'],
#                    ['X90p','Y90p','X90m','X90m']
#                ]

#cliffords_decomp=[
#                    ['X90p','X90m'],
#                    ['Y90p','X90p'],
#                    ['X90m','Y90m'],
#                    ['Y90p','Xp'],
#                    ['X90m'],
#                    ['X90p','Y90m','X90m'],
#                    ['Xp'],
#                    ['Y90m','X90m'],
#                    ['X90p','Y90m'],
#                    ['Y90m'],
#                    ['X90p'],
#                    ['X90p','Y90p','X90p'],
#                    ['Yp'],
#                    ['Y90m','X90p'],
#                    ['X90p','Y90p'],
#                    ['Y90m','Xp'],
#                    ['X90p','Yp'],
#                    ['X90p','Y90m','X90p'],
#                    ['Xp','Yp'],
#                    ['Y90p','X90m'],
#                    ['X90m','Y90p'],
#                    ['Y90p'],
#                    ['X90m','Yp'],
#                    ['X90p','Y90p','X90m']
#                    ]

cliffords_decomp=[
                    ['X90p','X90m'],
                    ['Y90p','X90p'],
                    ['X90m','Y90m'],
                    ['Y90p','X90p','X90p'],
                    ['X90m'],
                    ['X90p','Y90m','X90m'],
                    ['X90p','X90p'],
                    ['Y90m','X90m'],
                    ['X90p','Y90m'],
                    ['Y90m'],
                    ['X90p'],
                    ['X90p','Y90p','X90p'],
                    ['Y90p','Y90p'],
                    ['Y90m','X90p'],
                    ['X90p','Y90p'],
                    ['Y90m','X90p','X90p'],
                    ['X90p','Y90p','Y90p'],
                    ['X90p','Y90m','X90p'],
                    ['X90p','X90p','Y90p','Y90p'],
                    ['Y90p','X90m'],
                    ['X90m','Y90p'],
                    ['Y90p'],
                    ['X90m','Y90p','Y90p'],
                    ['X90p','Y90p','X90m']
                    ]
