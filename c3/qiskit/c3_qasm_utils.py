"""Convenience Module for creating different c3 components
c3_qasm_simulator
"""

from typing import List, Tuple
from qiskit import qobj

# Main C3 objects
from c3.c3objs import Quantity as Qty
from c3.parametermap import ParameterMap as Pmap
from c3.experiment import Experiment as Exp
from c3.system.model import Model as Mdl
from c3.generator.generator import Generator as Gnr

# Building blocks
import c3.generator.devices as devices
import c3.system.chip as chip
import c3.signal.pulse as pulse
import c3.signal.gates as gates
import c3.system.tasks as tasks

# Libs and helpers
import c3.libraries.algorithms as algorithms
import c3.libraries.hamiltonians as hamiltonians
import c3.libraries.fidelities as fidelities
import c3.libraries.envelopes as envelopes


def get_perfect_qubits(n_qubits: int) -> List[chip.Qubit]:
    """Instantiate and return a list of perfect C3 qubits

    Parameters
    ----------
    n_qubits : int
        Number of qubits to be returned

    Returns
    -------
    List[chip.Qubit]
        A list of perfect qubits
    """
    pass


def get_coupling_fc(n_qubits: int) -> List[chip.Coupling]:
    """Instantiate and return a fully connected coupling

    Parameters
    ----------
    n_qubits : int
        Number of qubits in the system

    Returns
    -------
    List[chip.Coupling]
        A fully connected coupling map
    """
    pass


def get_drives(n_qubits: int) -> List[chip.Drive]:
    """Instantiate and return drives for all qubits

    Parameters
    ----------
    n_qubits : int
        Number of qubits in the system

    Returns
    -------
    List[chip.Drive]
        List of drives for all the qubits
    """
    pass


def get_confusion_no_spam(n_qubits: int) -> tasks.ConfusionMatrix:
    """Return a dummy confusion matrix with no spam errors

    Parameters
    ----------
    n_qubits : int
        Number of qubits in the system

    Returns
    -------
    tasks.ConfusionMatrix
        No Spam errors Confusion Matrix
    """
    pass


def get_perfect_init_state() -> tasks.InitialiseGround:
    """Return an Initial Thermal State at 50 nK

    Returns
    -------
    tasks.InitialiseGround
        Thermal State at 50 nK
    """
    pass


def get_generator() -> Gnr:
    """C3 Model for an Generator

    Returns
    -------
    Gnr
        C3 Generator object
    """
    pass


def get_gate_set(n_qubits: int) -> List[gates.Instruction]:
    """List of single and 2-qubit gates for all qubits

    Parameters
    ----------
    n_qubits : int
        Number of qubits in the system

    Returns
    -------
    List[gates.Instruction]
        List of u3, cx, id and unitary gates for all qubits
    """
    pass


def get_opt_gates(n_qubits: int) -> List[str]:
    """Return the list of gates to optimize

    Parameters
    ----------
    n_qubits : int
        Number of qubits in system

    Returns
    -------
    List[str]
        Dummy list containing all gates
    """
    pass


def get_gateset_opt_map(n_qubits: int) -> List[List[Tuple]]:
    """Return parameter map to optimize

    Parameters
    ----------
    n_qubits : int
        Number of qubits in the system

    Returns
    -------
    List[List[Tuple]]
        Dummy list of gate parameters to optimize
    """
    pass


def get_sequence(instructions: qobj.QasmQobjExperiment.instructions) -> List[str]:
    """Return a sequence of gates from instructions

    Parameters
    ----------
    instructions : qobj.QasmQobjExperiment.instructions
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
