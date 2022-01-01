from typing import Optional
from qiskit.circuit import Gate
from qiskit.circuit.library import RXGate, RYGate, RZGate, CRXGate
from c3.libraries.constants import GATES
import numpy as np


class BaseC3QiskitGate(Gate):
    def __array__(self, dtype=complex):
        return GATES[self.name]


class RX90pGate(RXGate):
    def __init__(self, label: Optional[str] = None):
        super().__init__(np.pi / 2.0, label=label)
        self.name = "rx90p"


class RX90mGate(RXGate):
    def __init__(self, label: Optional[str] = None):
        super().__init__(-np.pi / 2.0, label=label)
        self.name = "rx90m"


class RXpGate(RXGate):
    def __init__(self, label: Optional[str] = None):
        super().__init__(np.pi, label=label)
        self.name = "rxp"


class RY90pGate(RYGate):
    def __init__(self, label: Optional[str] = None):
        super().__init__(np.pi / 2.0, label=label)
        self.name = "ry90p"


class RY90mGate(RYGate):
    def __init__(self, label: Optional[str] = None):
        super().__init__(-np.pi / 2.0, label=label)
        self.name = "ry90m"


class RYpGate(RYGate):
    def __init__(self, label: Optional[str] = None):
        super().__init__(np.pi, label=label)
        self.name = "ryp"


class RZ90pGate(RZGate):
    def __init__(self, label: Optional[str] = None):
        super().__init__(np.pi / 2.0, label=label)
        self.name = "rz90p"


class RZ90mGate(RZGate):
    def __init__(self, label: Optional[str] = None):
        super().__init__(-np.pi / 2.0, label=label)
        self.name = "rz90m"


class RZpGate(RZGate):
    def __init__(self, label: Optional[str] = None):
        super().__init__(np.pi, label=label)
        self.name = "rzp"


class CRXpGate(CRXGate):
    raise NotImplementedError(
        "Not implemented due to inconsistent matrix representation in C3 and Qiskit"
    )


class CRGate(BaseC3QiskitGate):
    pass


class CR90Gate(BaseC3QiskitGate):
    pass
