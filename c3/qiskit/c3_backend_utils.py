"""Convenience Module for creating different c3_backend
"""
from typing import Dict
import numpy as np
import tensorflow as tf
import math

GATE_MAP = {
    "x": "rxp",
    "y": "ryp",
    "z": "rzp",
    "cx": "crxp",
    "cz": "crzp",
    "I": "id",
    "u0": "id",
    "id": "id",
    "iSwap": "iswap",
}

PARAMETER_MAP = {np.pi / 2: "90p", np.pi: "p", -np.pi / 2: "90m", -np.pi: "m"}


def make_gate_str(instruction: dict, gate_name: str) -> str:
    """Make C3 style gate name string

    Parameters
    ----------
    instruction : Dict[str, Any]
        A dict in OpenQasm instruction format ::

            {"name": "rx", "qubits": [0], "params": [1.57]}
    gate_name : str
        C3 style gate names

    Returns
    -------
    str
        C3 style gate name + qubit string ::

            {"name": "rx", "qubits": [0], "params": [1.57]} -> rx90p[0]
    """
    qubits = instruction.qubits  # type: ignore
    gate_str = gate_name + str(qubits)
    return gate_str


def get_init_ground_state(n_qubits: int, n_levels: int) -> tf.Tensor:
    """Return a perfect ground state

    Parameters
    ----------
    n_qubits : int
        Number of qubits in the system

    n_levels : int
        Number of levels for each qubit

    Returns
    -------
    tf.Tensor
        Tensor array of ground state
        shape(m^n, 1), dtype=complex128
        m = no of qubit levels
        n = no of qubits
    """
    psi_init = [[0] * (int)(math.pow(n_levels, n_qubits))]
    psi_init[0][0] = 1
    init_state = tf.transpose(tf.constant(psi_init, tf.complex128))

    return init_state


def flip_labels(counts: Dict[str, int]) -> Dict[str, int]:
    """Flip C3 qubit labels to match Qiskit qubit indexing

    Parameters
    ----------
    counts : Dict[str, int]
        OpenQasm 2.0 result counts with original C3 style
        qubit indices

    Returns
    -------
    Dict[str, int]
        OpenQasm 2.0 result counts with Qiskit style labels

    Note
    ----
    Basis vector ordering in Qiskit

    Qiskit uses a slightly different ordering of the qubits compared to
    what is seen in Physics textbooks. In qiskit, the qubits are represented from
    the most significant bit (MSB) on the left to the least significant bit (LSB)
    on the right (big-endian). This is similar to bitstring representation
    on classical computers, and enables easy conversion from bitstrings to
    integers after measurements are performed.

    More details:
    https://qiskit.org/documentation/tutorials/circuits/3_summary_of_quantum_operations.html#Basis-vector-ordering-in-Qiskit

    """

    # TODO: https://github.com/q-optimize/c3/issues/58
    labels_flipped_counts = {}
    for key, value in counts.items():
        key_bin = bin(int(key, 0))
        key_bin_rev = "0b" + key_bin[:1:-1]
        key_rev = hex(int(key_bin_rev, 0))
        labels_flipped_counts[key_rev] = value
    return labels_flipped_counts
