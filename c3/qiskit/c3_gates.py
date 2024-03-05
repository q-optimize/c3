"""Library for interoperability of c3 gates with qiskit
"""
from typing import Iterable, List, Optional
from qiskit.circuit.library import UnitaryGate
from qiskit.circuit import Gate
from qiskit.circuit.library import RXGate, RYGate, RZGate, CRXGate
from c3.libraries.constants import GATES
import numpy as np
import warnings


class RX90pGate(RXGate):
    """90 degree rotation around X axis in the positive direction"""

    def __init__(self, label: Optional[str] = None):
        super().__init__(np.pi / 2.0, label=label)
        self.name = "rx90p"


class RX90mGate(RXGate):
    """90 degree rotation around X axis in the negative direction"""

    def __init__(self, label: Optional[str] = None):
        super().__init__(-np.pi / 2.0, label=label)
        self.name = "rx90m"


class RXpGate(RXGate):
    """180 degree rotation around X axis"""

    def __init__(self, label: Optional[str] = None):
        super().__init__(np.pi, label=label)
        self.name = "rxp"


class RY90pGate(RYGate):
    """90 degree rotation around Y axis in the positive direction"""

    def __init__(self, label: Optional[str] = None):
        super().__init__(np.pi / 2.0, label=label)
        self.name = "ry90p"


class RY90mGate(RYGate):
    """90 degree rotation around Y axis in the negative direction"""

    def __init__(self, label: Optional[str] = None):
        super().__init__(-np.pi / 2.0, label=label)
        self.name = "ry90m"


class RYpGate(RYGate):
    """180 degree rotation around Y axis"""

    def __init__(self, label: Optional[str] = None):
        super().__init__(np.pi, label=label)
        self.name = "ryp"


class RZ90pGate(RZGate):
    """90 degree rotation around Z axis in the positive direction"""

    def __init__(self, label: Optional[str] = None):
        super().__init__(np.pi / 2.0, label=label)
        self.name = "rz90p"


class RZ90mGate(RZGate):
    """90 degree rotation around Z axis in the negative direction"""

    def __init__(self, label: Optional[str] = None):
        super().__init__(-np.pi / 2.0, label=label)
        self.name = "rz90m"


class RZpGate(RZGate):
    """180 degree rotation around Z axis"""

    def __init__(self, label: Optional[str] = None):
        super().__init__(np.pi, label=label)
        self.name = "rzp"


class CRXpGate(CRXGate):
    def __init__(self):
        raise NotImplementedError(
            "Not implemented due to inconsistent matrix representation in C3 and Qiskit"
        )


class CRGate(UnitaryGate):
    """Cross Resonance Gate

    Warnings
    ---------
    This is not equivalent to the RZX(pi/2) gate in qiskit
    """

    def __init__(self, label: Optional[str] = None):
        warnings.warn("This is not equivalent to the RZX(pi/2) gate in qiskit")
        super().__init__(data=GATES["cr"], label=label)
        self.name = "cr"


class CR90Gate(UnitaryGate):
    """Cross Resonance 90 degree gate

    Warnings
    ---------
    This is not equivalent to the RZX(pi/2) gate in qiskit
    """

    def __init__(self, label: Optional[str] = None):
        warnings.warn("This is not equivalent to the RZX(pi/2) gate in qiskit")
        super().__init__(data=GATES["cr90"], label=label)
        self.name = "cr90"


class SetParamsGate(Gate):
    """Gate for setting parameter values through qiskit interface. This gate is only
    processed when it is the last gate in the circuit, otherwise it throws a KeyError.
    The qubit target for the gate can be any valid qubit in the circuit, this argument
    is currently ignored and not processed by the backend

    These parameters should be supplied as a list with the first item a list of
    Quantity objects converted to a Dict of Python primitives and the second item an
    opt_map with the proper list nesting. For example: ::

        amp = Qty(value=0.8, min_val=0.2, max_val=1, unit="V")
        opt_map = [[["rx90p[0]", "d1", "gaussian", "amp"]]]
        param_gate = SetParamsGate(params=[[amp.asdict()], opt_map])
    """

    def __init__(self, params: List) -> None:
        name = "param_update"
        num_qubits = 1
        label = None
        super().__init__(name, num_qubits, params, label)

    def validate_parameter(self, parameter: Iterable) -> Iterable:
        """Override the default validation in Gate to allow arbitrary lists
        ----------
        parameter : Iterable
            The waveform as a nested list
        Returns
        -------
        Iterable
            The same waveform
        """
        # TODO implement sanitation/validation as required
        return parameter
