"""
testing module quick setup class
"""

from c3.experiment import Experiment

exp = Experiment()
exp.quick_setup("test/quickstart.hjson")
pmap = exp.pmap
model = pmap.model
generator = pmap.generator


def test_exp_quick_setup_freqs() -> None:
    """
    Test the quick setup.
    """
    qubit_freq = model.subsystems["Q1"].params["freq"].get_value()
    gate = pmap.instructions["X90p:Id"]
    carrier_freq = gate.comps["d1"]["carrier"].params["freq"].get_value()
    offset = gate.comps["d1"]["gaussian"].params["freq_offset"].get_value()
    assert qubit_freq == carrier_freq + offset
