"""Convenience Module for creating different c3 components
c3_qasm_simulator
"""
from typing import List
import tensorflow as tf
import math


def get_sequence(instructions: dict) -> List[str]:
    """Return a sequence of gates from instructions

    Parameters
    ----------
    instructions : dict
        Instructions from the qasm experiment

    Returns
    -------
    List[str]
        List of gates
    """
    # TODO conditional
    # conditional = getattr(instructions, "conditional", None)  # noqa

    # TODO unitary

    # TODO U, u3

    # TODO CX, cx

    # TODO id, u0

    # TODO reset

    # TODO barrier

    # TODO measure

    # TODO binary function

    # TODO raise C3QiskitError if unknown instruction
    pass


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
