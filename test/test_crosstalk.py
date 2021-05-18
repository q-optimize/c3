import pytest

import numpy as np


@pytest.mark.unit
def test_crosstalk(get_xtalk_pmap, get_test_signal) -> None:
    xtalk = get_xtalk_pmap.generator.devices["crosstalk"]
    new_sig = xtalk.process(signal=get_test_signal)
    assert new_sig == get_test_signal


@pytest.mark.unit
def test_crosstalk_flip(get_xtalk_pmap, get_test_signal) -> None:
    xtalk = get_xtalk_pmap.generator.devices["crosstalk"]
    xtalk.params["crosstalk_matrix"].set_value([[0, 1], [1, 0]])
    new_sig = xtalk.process(signal=get_test_signal)
    assert (new_sig["TC2"]["values"].numpy() == np.linspace(0, 100, 101)).all()
    assert (new_sig["TC1"]["values"].numpy() == np.linspace(100, 200, 101)).all()


@pytest.mark.unit
def test_crosstalk_mix(get_xtalk_pmap, get_test_signal) -> None:
    xtalk = get_xtalk_pmap.generator.devices["crosstalk"]
    xtalk.params["crosstalk_matrix"].set_value([[0.5, 0.5], [0.5, 0.5]])
    new_sig = xtalk.process(signal=get_test_signal)
    assert (new_sig["TC2"]["values"].numpy() == new_sig["TC1"]["values"].numpy()).all()


@pytest.mark.unit
def test_crosstalk_set_get_parameters(get_xtalk_pmap) -> None:
    get_xtalk_pmap.set_parameters(
        [[[1, 1], [1, 1]]], [[["crosstalk", "crosstalk_matrix"]]]
    )
    assert (
        get_xtalk_pmap.get_parameters()[0].get_value().numpy() == [[1, 1], [1, 1]]
    ).all()
