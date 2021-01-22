"""Convenience Module for creating different c3_backend
"""
from typing import Any, Dict, List
import tensorflow as tf
import math
from .c3_exceptions import C3QiskitError


def get_sequence(instructions: List[Dict[Any, Any]]) -> List[str]:
    """Return a sequence of gates from instructions

    Parameters
    ----------
    instructions : List[dict]
        Instructions from the qasm experiment

    Returns
    -------
    List[str]
        List of gates
    """

    sequence = []

    for instruction in instructions:

        # Conditional operations are not supported
        conditional = getattr(instructions, "conditional", None)  # noqa
        if conditional is not None:
            raise C3QiskitError("C3 Simulator does not support conditional operations")

        # reset is not supported
        if instruction.name == "reset":  # type: ignore
            raise C3QiskitError("C3 Simulator does not support qubit reset")

        # binary functions are not supported
        elif instruction.name == "bfunc":  # type: ignore
            raise C3QiskitError("C3 Simulator does not support binary functions")

        # barrier is implemented internally through Identity gates
        elif instruction.name == "barrier":  # type: ignore
            pass

        # TODO X
        elif instruction.name == "x":  # type: ignore
            pass

        # TODO U, u3
        elif instruction.name in ("U", "u3"):  # type: ignore
            pass

        # TODO CX, cx
        elif instruction.name in ("CX", "cx"):  # type: ignore
            pass

        # id, u0 implemented internally
        elif instruction.name in ("id", "u0"):  # type: ignore
            pass

        # TODO measure using evaluate()
        elif instruction.name == "measure":  # type: ignore
            pass

        # raise C3QiskitError if unknown instruction
        else:
            raise C3QiskitError(
                "Encountered unknown operation {}".format(instruction.name)  # type: ignore
            )
    sequence = ["X90p:Id"]
    return sequence


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
