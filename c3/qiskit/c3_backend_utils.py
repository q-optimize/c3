"""Convenience Module for creating different c3_backend
"""
from typing import Dict, List
import tensorflow as tf
import math
from .c3_exceptions import C3QiskitError

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


def pad_gate_name(gate_name: str, qubits: List[int], n_qubits: int) -> str:
    """Pad gate name with Identity gates in correct indices

    Parameters
    ----------
    gate_name : str
        A C3 compatible gate name
    qubits : List[int]
        Indices to apply gate
    n_qubits : int
        Total number of qubits in the device

    Returns
    -------
    str
        Identity padded gate name, eg ::

            pad_gate_name("CCRX90p", [1, 2, 3], 5) -> 'Id:CCRX90p:Id'
            pad_gate_name("RX90p", [0], 5) -> 'RX90p:Id:Id:Id:Id'
    """

    # TODO (check) Assumption control and action qubits next to each other
    gate_names = ["Id"] * (n_qubits - (len(qubits) - 1))
    gate_names[qubits[0]] = gate_name
    padded_gate_str = ":".join(gate_names)
    return padded_gate_str


def get_sequence(instructions: List, n_qubits: int) -> List[str]:
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

    n_qubits: int
        Number of qubits in the device config

    Returns
    -------
    List[str]
        List of gates, for example::

        sequence = ["RX90p:Id", "Id:RX90p", "CR90"]

    """

    sequence = []

    for instruction in instructions:

        # TODO Check if gate is possible from device_config
        # TODO parametric gates

        iname = instruction.name
        # Conditional operations are not supported
        conditional = getattr(instructions, "conditional", None)  # noqa
        if conditional is not None:
            raise C3QiskitError("C3 Simulator does not support conditional operations")

        # reset, binary functions is not supported
        elif iname in ["reset", "bfunc"]:
            raise C3QiskitError("C3 Simulator does not support {}".format(iname))

        # barrier is implemented internally through Identity gates
        elif iname == "barrier":
            pass

        # TODO U, u3
        elif iname in ("U", "u3"):
            raise C3QiskitError("U3 gates are not yet implemented")

        # measure implemented outside sequences
        elif iname == "measure":
            pass

        elif iname in ["rx", "ry", "rz", "rzx"]:
            pass

        elif iname in GATE_MAP.keys():
            gate_name = GATE_MAP[iname]
            qubits = instruction.qubits
            gate_str = gate_name + str(qubits)
            sequence.append(gate_str)

        # raise C3QiskitError if unknown instruction
        else:
            raise C3QiskitError("Encountered unknown operation {}".format(iname))

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
    labels_flipped_counts = {}
    for key, value in counts.items():
        key_bin = bin(int(key, 0))
        key_bin_rev = "0b" + key_bin[:1:-1]
        key_rev = hex(int(key_bin_rev, 0))
        labels_flipped_counts[key_rev] = value
    return labels_flipped_counts
