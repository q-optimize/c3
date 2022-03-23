"""
testing module quick setup class
"""

import numpy as np
import pytest
import pickle
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

@pytest.mark.integration
def test_generator() -> None:
    gen_signal = pmap.generator.generate_signals(pmap.instructions["rx90p[0]"])
    with open("test/quick_data.pickle", "rb") as quickfile:
        test_data = pickle.load(quickfile)
    np.testing.assert_allclose(gen_signal["d1"]["ts"], test_data["d1"]["ts"])
    np.testing.assert_allclose(
        actual=gen_signal["d1"]["values"].numpy(),
        desired=test_data["d1"]["values"].numpy(),
    )
