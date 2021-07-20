"""
testing module quick setup class
"""

import pytest
from c3.experiment import Experiment

exp = Experiment()
exp.load_quick_setup("test/quickstart.hjson")
pmap = exp.pmap
model = pmap.model
generator = pmap.generator


@pytest.mark.integration
def test_exp_quick_setup_freqs() -> None:
    """
    Test the quick setup.
    """
    print(pmap.instructions.keys())
    qubit_freq = model.subsystems["Q1"].params["freq"].get_value()
    gate = pmap.instructions["rx90p[0]"]
    carrier_freq = gate.comps["d1"]["carrier"].params["freq"].get_value()
    offset = gate.comps["d1"]["gaussian"].params["freq_offset"].get_value()
    assert qubit_freq == carrier_freq + offset
