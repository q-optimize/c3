"""
integration testing module for C1 optimization through two-qubits example
"""

import pickle
import numpy as np
import pytest
from numpy.testing import assert_array_almost_equal as almost_equal

# Libs and helpers
from c3.libraries.propagation import rk4

with open("test/two_qubit_data.pickle", "rb") as filename:
    test_data = pickle.load(filename)


@pytest.mark.tensorflow
@pytest.mark.unit
def test_signals(get_two_qubit_chip) -> None:
    """Compare generated control signals to stored examples."""
    exp = get_two_qubit_chip
    pmap = exp.pmap
    gen_signal = pmap.generator.generate_signals(pmap.instructions["rx90p[0]"])
    ts = gen_signal["d1"]["ts"]
    np.testing.assert_allclose(ts, test_data["ts"])
    np.testing.assert_allclose(
        actual=gen_signal["d1"]["values"].numpy(),
        desired=test_data["signal"]["d1"]["values"].numpy(),
    )


@pytest.mark.unit
def test_hamiltonians(get_two_qubit_chip) -> None:
    """Compare Hamilonians"""
    model = get_two_qubit_chip.pmap.model
    hdrift, hks = model.get_Hamiltonians()
    assert (hdrift.numpy() - test_data["hdrift"].numpy() < 1).any()
    for key in hks:
        almost_equal(hks[key], test_data["hks"][key])


@pytest.mark.tensorflow
@pytest.mark.integration
def test_propagation(get_two_qubit_chip) -> None:
    """Test that result of the propagation code does not change."""
    exp = get_two_qubit_chip
    pmap = exp.pmap
    result = exp.propagation(pmap.model, pmap.generator, pmap.instructions["rx90p[0]"])
    propagator = result["U"]
    almost_equal(propagator, test_data["propagator"])


@pytest.mark.unit
def test_init_point(get_OC_optimizer) -> None:
    """Check that a previous best point can be loaded as an initial point."""
    opt = get_OC_optimizer
    opt.load_best("test/best_point_open_loop.c3log")


@pytest.mark.slow
@pytest.mark.tensorflow
@pytest.mark.optimizers
@pytest.mark.integration
def test_optim(get_OC_optimizer) -> None:
    """
    check if optimization result is below 1e-1
    """
    opt = get_OC_optimizer
    opt.optimize_controls()
    assert opt.current_best_goal < 0.1


@pytest.mark.tensorflow
@pytest.mark.integration
def test_rk4(get_two_qubit_chip) -> None:
    """Testing that RK4 exists and runs."""
    exp = get_two_qubit_chip
    pmap = exp.pmap
    exp.set_prop_method(rk4)
    exp.propagation(pmap.model, pmap.generator, pmap.instructions["rx90p[0]"])
