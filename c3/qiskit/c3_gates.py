from typing import Optional
from qiskit.circuit import Gate
from qiskit import QuantumCircuit
from c3.libraries.constants import GATES
import numpy as np


class BaseC3QiskitGate(Gate):
    def __array__(self, dtype=complex):
        return GATES[self.name]


class RX90pGate(BaseC3QiskitGate):
    def __init__(self, label: Optional[str] = None):
        super().__init__("rx90p", 1, [], label=label)

    def _define(self):
        qc = QuantumCircuit(1)
        qc.rx(np.pi / 2.0, [0])
        self.definition = qc


class RX90mGate(BaseC3QiskitGate):
    pass


class RXpGate(BaseC3QiskitGate):
    pass


class RY90pGate(BaseC3QiskitGate):
    pass


class RY90mGate(BaseC3QiskitGate):
    pass


class RYpGate(BaseC3QiskitGate):
    pass


class RZ90pGate(BaseC3QiskitGate):
    pass


class RZ90mGate(BaseC3QiskitGate):
    pass


class RZpGate(BaseC3QiskitGate):
    pass


class CRXpGate(BaseC3QiskitGate):
    pass


class CRGate(BaseC3QiskitGate):
    pass


class CR90Gate(BaseC3QiskitGate):
    pass
