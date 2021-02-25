import pytest

import numpy as np
import tensorflow as tf

from c3.generator.devices import Crosstalk
from c3.generator.generator import Generator
from c3.c3objs import Quantity as Qty
from c3.parametermap import ParameterMap


xtalk = Crosstalk(
    name="crosstalk",
    channels=["TC1", "TC2"],
    crosstalk_matrix=Qty(
        value=[[1, 0], [0, 1]],
        min_val=[[0, 0], [0, 0]],
        max_val=[[1, 1], [1, 1]],
        unit="",
    ),
)

signal = {
    "TC1": {"values": tf.linspace(0, 100, 101)},
    "TC2": {"values": tf.linspace(100, 200, 101)},
}

gen = Generator(devices={"crosstalk": xtalk})
pmap = ParameterMap(generator=gen)
pmap.set_opt_map([[["crosstalk", "crosstalk_matrix"]]])


@pytest.mark.unit
def test_crosstalk() -> None:
    new_sig = xtalk.process(signal=signal)
    assert new_sig == signal


@pytest.mark.unit
def test_crosstalk_flip() -> None:
    xtalk.params["crosstalk_matrix"].set_value([[0, 1], [1, 0]])
    new_sig = xtalk.process(signal=signal)
    assert (new_sig["TC2"]["values"].numpy() == np.linspace(0, 100, 101)).all()
    assert (new_sig["TC1"]["values"].numpy() == np.linspace(100, 200, 101)).all()


@pytest.mark.unit
def test_crosstalk_mix() -> None:
    xtalk.params["crosstalk_matrix"].set_value([[0.5, 0.5], [0.5, 0.5]])
    new_sig = xtalk.process(signal=signal)
    assert (new_sig["TC2"]["values"].numpy() == new_sig["TC1"]["values"].numpy()).all()


@pytest.mark.unit
def test_crosstalk_set_get_parameters() -> None:
    pmap.set_parameters([[1, 1], [1, 1]], [[["crosstalk", "crosstalk_matrix"]]])
    assert (pmap.get_parameters()[0].get_value().numpy() == [[1, 1], [1, 1]]).all()
