"""Convenience Module for creating different c3_backend
"""
from typing import List
import tensorflow as tf
import math
from .c3_exceptions import C3QiskitError


def get_sequence(instructions: List) -> List[str]:
    """Return a sequence of gates from instructions

    Parameters
    ----------
    instructions : List[dict]
        Instructions from the qasm experiment, for example::

        instructions: [
                {"name": "u1", "qubits": [0], "params": [0.4]},
                {"name": "u2", "qubits": [0], "params": [0.4,0.2]},
                {"name": "u3", "qubits": [0], "params": [0.4,0.2,-0.3]},
                {"name": "snapshot", "label": "snapstate1", "snapshot_type": "statevector"},
                {"name": "cx", "qubits": [0,1]},
                {"name": "barrier", "qubits": [0]},
                {"name": "measure", "qubits": [0], "register": [1], "memory": [0]},
                {"name": "u2", "qubits": [0], "params": [0.4,0.2], "conditional": 2}
            ]

    Returns
    -------
    List[str]
        List of gates, for example::

        sequence = ["RX90p:Id", "Id:RX90p", "CR90"]

    """

    sequence = []

    for instruction in instructions:

        # Conditional operations are not supported
        conditional = getattr(instructions, "conditional", None)  # noqa
        if conditional is not None:
            raise C3QiskitError("C3 Simulator does not support conditional operations")

        # reset is not supported
        if instruction.name == "reset":
            raise C3QiskitError("C3 Simulator does not support qubit reset")

        # binary functions are not supported
        elif instruction.name == "bfunc":
            raise C3QiskitError("C3 Simulator does not support binary functions")

        # barrier is implemented internally through Identity gates
        elif instruction.name == "barrier":
            pass

        # TODO scalable way to name and assign X gate in multi qubit systems
        elif instruction.name == "x":
            if instruction.qubits[0] == 0:
                sequence.append("RX90p:Id")
            elif instruction.qubits[0] == 1:
                sequence.append("Id:RX90p")
            else:
                raise C3QiskitError(
                    "Gate {0} on qubit {1} not possible".format(
                        instruction.name, instruction.qubits[0]
                    )
                )

        # TODO U, u3
        elif instruction.name in ("U", "u3"):
            raise C3QiskitError("U3 gates are not yet implemented in C3 Simulator")

        # TODO scalable way to name and assign CX, cx gate in multi qubit systems
        elif instruction.name in ("CX", "cx"):
            if instruction.qubits == [0, 1]:
                sequence.append("CR90")
            else:
                raise C3QiskitError(
                    "Gate {0} on qubits {1} not possible".format(
                        instruction.name, instruction.qubits
                    )
                )

        # TODO scalable way to assign CZ gates and inverse control
        elif instruction.name in ("CZ", "cz"):
            if instruction.qubits == [0, 1]:
                sequence.append("CZ")
            else:
                raise C3QiskitError(
                    "Gate {0} on qubits {1} not possible".format(
                        instruction.name, instruction.qubits
                    )
                )

        # id, u0 implemented internally
        elif instruction.name in ("id", "u0"):
            pass

        # measure implemented outside sequences
        elif instruction.name == "measure":
            pass

        # raise C3QiskitError if unknown instruction
        else:
            raise C3QiskitError(
                "Encountered unknown operation {}".format(instruction.name)
            )

    # TODO implement padding
    # TODO fix gate naming bugs
    sequence = [
        "RX90p:Id:Id:Id:Id:Id",
        "Id:RX90p:Id:Id:Id:Id",
        "CR90:Id:Id:Id:Id",
        "RX90p:RX90p:Id:Id:Id:Id",
        "RX90p:Id:Id:Id:Id:Id",
    ]
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
