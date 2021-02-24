"""All physical constants used in other code."""

import numpy as np

global kb, h, hbar


kb = 1.380649e-23
h = 6.62607015e-34
hbar = 1.054571817e-34

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

GATES = {
    "id": np.array([[1, 0], [0, 1]], dtype=np.complex128),
    "rx90p": np.array([[1, -1j], [-1j, 1]], dtype=np.complex128) / np.sqrt(2),
    "rx90m": np.array([[1, 1j], [1j, 1]], dtype=np.complex128) / np.sqrt(2),
    "rxp": np.array([[0, -1j], [-1j, 0]], dtype=np.complex128),
    "ry90p": np.array([[1, -1], [1, 1]], dtype=np.complex128) / np.sqrt(2),
    "ry90m": np.array([[1, 1], [-1, 1]], dtype=np.complex128) / np.sqrt(2),
    "ryp": np.array([[0, -1], [1, 0]], dtype=np.complex128),
    "rz90p": np.array([[1 - 1j, 0], [0, 1 + 1j]], dtype=np.complex128) / np.sqrt(2),
    "rz90m": np.array([[1 + 1j, 0], [0, 1 - 1j]], dtype=np.complex128) / np.sqrt(2),
    "rzp": np.array([[-1.0j, 0], [0, 1.0j]], dtype=np.complex128),
    "crxp": np.array(
        [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, -1j], [0, 0, -1j, 0]],
        dtype=np.complex128,
    ),
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
        [[1, 0, 0, 0], [0, 0, 1j, 0], [0, 1j, 0, 0], [0, 0, 0, 1]], dtype=np.complex128
    ),
}
