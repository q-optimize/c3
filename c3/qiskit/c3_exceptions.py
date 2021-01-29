"""
Exception for errors raised by Basic Aer.
"""

from qiskit.exceptions import QiskitError


class C3QiskitError(QiskitError):
    """Base class for errors raised by C3 Qiskit Simulator."""

    def __init__(self, *message):
        """Set the error message."""
        super().__init__(*message)
        self.message = " ".join(message)

    def __str__(self):
        """Return the message."""
        return repr(self.message)
