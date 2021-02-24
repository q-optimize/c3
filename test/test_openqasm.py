"""
testing module for QASM instructions
"""

import pytest
from c3.experiment import Experiment

exp = Experiment()
exp.quick_setup("test/quickstart.hjson")
exp.enable_qasm()

sequence = [
    {"name": "rx90p", "qubits": [0]},
    {"name": "VZ", "qubits": [0], "params": [0.123]},
    {"name": "VZ", "qubits": [1], "params": [2.31]},
]
exp.set_opt_gates([str(("rx90p", [0]))])
exp.compute_propagators()


@pytest.mark.integration
def test_qasm_sequence() -> None:
    """
    Run instructions in a qasm format.
    """
    exp.evaluate([sequence])
