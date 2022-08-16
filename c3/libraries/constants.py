"""All physical constants used in other code."""

import numpy as np

kb = 1.380649e-23
h = 6.62607015e-34
hbar = 1.054571817e-34
q_e = 1.602176634e-19  # electron charge
twopi = 6.2831853071795864769252867665590057683943387987502116419498891846

PREFIXES = {
    "K": 1e3,
    "M": 1e6,
    "G": 1e9,
    "T": 1e12,
    "m": 1e-3,
    "µ": 1e-6,
    "n": 1e-9,
    "p": 1e-12,
}

# Pauli matrices
Id = np.array([[1, 0], [0, 1]], dtype=np.complex128)
X = np.array([[0, 1], [1, 0]], dtype=np.complex128)
Y = np.array([[0, -1j], [1j, 0]], dtype=np.complex128)
Z = np.array([[1, 0], [0, -1]], dtype=np.complex128)

PAULIS = {"X": X, "Y": Y, "Z": Z, "Id": Id}

ISWAP = np.array(
    [[1, 0, 0, 0], [0, 0, 1j, 0], [0, 1j, 0, 0], [0, 0, 0, 1]], dtype=np.complex128
)
ISWAP3 = np.array(
    [
        [1, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 1j, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 1j, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 1, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0],
    ],
    dtype=np.complex128,
)

CCZ = np.diag([1, 1, 1, 1, 1, 1, 1, -1])

GATES = {
    "id": np.array([[1, 0], [0, 1]], dtype=np.complex128),
    "rx90p": np.array([[1, -1j], [-1j, 1]], dtype=np.complex128) / np.sqrt(2),
    "rx90m": np.array([[1, 1j], [1j, 1]], dtype=np.complex128) / np.sqrt(2),
    "rxp": np.array([[0, -1j], [-1j, 0]], dtype=np.complex128),
    "ry90p": np.array([[1, -1], [1, 1]], dtype=np.complex128) / np.sqrt(2),
    "ry90m": np.array([[1, 1], [-1, 1]], dtype=np.complex128) / np.sqrt(2),
    "ryp": np.array([[0, -1], [1, 0]], dtype=np.complex128),
    "x": X,
    "y": Y,
    "h": (1 / np.sqrt(2)) * np.array([[1, 1], [1, -1]]),
    "rz90p": np.array([[1 - 1j, 0], [0, 1 + 1j]], dtype=np.complex128) / np.sqrt(2),
    "rz90m": np.array([[1 + 1j, 0], [0, 1 - 1j]], dtype=np.complex128) / np.sqrt(2),
    "rzp": np.array([[-1.0j, 0], [0, 1.0j]], dtype=np.complex128),
    "crxp": np.array(
        [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, -1j], [0, 0, -1j, 0]],
        dtype=np.complex128,
    ),
    # What is the meaning of this gate?
    "crzp": np.array(
        [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, -1j, 0], [0, 0, 0, 1j]], dtype=np.complex128
    ),
    "cr": np.array(
        [[0, -1j, 0, 0], [-1j, 0, 0, 0], [0, 0, 0, 1j], [0, 0, 1j, 0]],
        dtype=np.complex128,
    ),
    "cr90": np.array(
        [[1, -1j, 0, 0], [-1j, 1, 0, 0], [0, 0, 1, 1j], [0, 0, 1j, 1]],
        dtype=np.complex128,
    )
    / np.sqrt(2),
    "iswap": np.array(
        [[1, 0, 0, 0], [0, 0, -1j, 0], [0, -1j, 0, 0], [0, 0, 0, 1]],
        dtype=np.complex128,
    ),
    "iswap90": np.array(
        [[0, 0, 0, 0], [0, 1, -1j, 0], [0, -1j, 1, 0], [0, 0, 0, 0]],
        dtype=np.complex128,
    )
    / np.sqrt(2)
    + np.array(
        [[1, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 1]], dtype=np.complex128
    ),
    "cz": np.diag(np.array([1, 1, 1, -1], dtype=np.complex128)),
    "ccz": np.diag(np.array([1, 1, 1, 1, 1, 1, 1, -1], dtype=np.complex128)),
    "cx": np.array(
        [[1, 0, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0], [0, 1, 0, 0]], dtype=np.complex128
    ),
}

x90p = GATES["rx90p"]
y90p = GATES["ry90p"]
x90m = GATES["rx90m"]
y90m = GATES["ry90m"]

CLIFFORDS = {
    "C1": x90m @ x90p,
    "C2": x90p @ y90p,
    "C3": y90m @ x90m,
    "C4": x90p @ x90p @ y90p,
    "C5": x90m,
    "C6": x90m @ y90m @ x90p,
    "C7": x90p @ x90p,
    "C8": x90m @ y90m,
    "C9": y90m @ x90p,
    "C10": y90m,
    "C11": x90p,
    "C12": x90p @ y90p @ x90p,
    "C13": y90p @ y90p,
    "C14": x90p @ y90m,
    "C15": y90p @ x90p,
    "C16": x90p @ x90p @ y90m,
    "C17": y90p @ y90p @ x90p,
    "C18": x90p @ y90m @ x90p,
    "C19": y90p @ y90p @ x90p @ x90p,
    "C20": x90m @ y90p,
    "C21": y90p @ x90m,
    "C22": y90p,
    "C23": y90p @ y90p @ x90m,
    "C24": x90m @ y90p @ x90p,
}
