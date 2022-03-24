"""Unit tests for Quantity class"""

import hjson
import numpy as np
import pytest
from c3.c3objs import Quantity, hjson_encode

amp = Quantity(value=0.0, min_val=-1.0, max_val=+1.0, unit="V")
amp_dict = {
    "value": 0.0,
    "min_val": -1.0,
    "max_val": 1.0,
    "unit": "V",
    "symbol": "\\alpha",
}

freq = Quantity(
    value=5.6e9, min_val=5.595e9, max_val=5.605e9, unit="Hz 2pi", symbol="\\omega"
)
freq_dict = {
    "value": 5.6e9,
    "min_val": 5.595e9,
    "max_val": 5.605e9,
    "unit": "Hz 2pi",
    "symbol": "\\omega",
}

gate_time = Quantity(
    value=5.3246e-9, min_val=2e-9, max_val=10e-9, unit="s", symbol=r"t_g"
)

matrix = Quantity(
    value=[[0, 1], [1, 0]],
    min_val=[[0, 0], [0, 0]],
    max_val=[[1, 1], [1, 1]],
    unit="",
    symbol=r"M",
)


@pytest.mark.unit
def test_qty_2pi() -> None:
    assert freq.asdict() == freq_dict


@pytest.mark.unit
def test_qty_set_2pi() -> None:
    freq.set_value(5.602e9)
    assert freq.get_value() - 5.602e9 * 2 * np.pi < 1e-8


@pytest.mark.unit
def test_qty_asdict() -> None:
    assert amp.asdict() == amp_dict


@pytest.mark.unit
def test_qty_write_cfg() -> None:
    print(hjson.dumps(amp.asdict(), default=hjson_encode))


@pytest.mark.unit
def test_qty_read_cfg() -> None:
    assert Quantity(**amp_dict).asdict() == amp.asdict()


@pytest.mark.unit
def test_qty_str() -> None:
    assert str(gate_time) == "5.325 ns "


@pytest.mark.unit
def test_qty_set() -> None:
    gate_time.set_value(7e-9)
    assert gate_time.get_value() - 7e-9 < 1e-15


@pytest.mark.unit
def test_qty_max() -> None:
    gate_time.set_opt_value(1.0)
    assert gate_time.get_value() - 10e-9 < 1e-15


@pytest.mark.unit
def test_qty_min() -> None:
    gate_time.set_opt_value(-1.0)
    assert gate_time.get_value() - 2e-9 < 1e-15


@pytest.mark.unit
def test_qty_get_opt() -> None:
    gate_time.set_value(6e-9)
    assert gate_time.get_opt_value() < 1e-15


@pytest.mark.unit
def test_qty_matrix_str() -> None:
    assert str(matrix) == "0.000  1.000  1.000  0.000  "


@pytest.mark.unit
def test_qty_matrix_set() -> None:
    matrix.set_value([[1.0, 0.0], [0.0, 1.0]])
    assert (matrix.numpy() == [[1, 0], [0, 1]]).all()


@pytest.mark.unit
def test_qty_matrix_set_opt() -> None:
    assert (matrix.get_opt_value() == [1.0, -1.0, -1.0, 1.0]).all()


@pytest.mark.unit
def test_qty_np_conversions() -> None:
    a = Quantity(value=3, unit="unit")
    assert repr(a) == "3.000 unit"
    assert np.mod(a, 2) == 1.0
    assert type(a.numpy()) is np.float64 or type(a.numpy()) is np.ndarray
    assert a + a == 6
    np.array([a])  # test conversion
    np.array(a)
    float(a)
    assert np.mod([a], 2) == np.array([[1.0]])
    assert list(a) == [3.0]

    b = Quantity(np.array([0.0000001, 0.00001]))
    np.array([b])

    c = Quantity([0, 0.1], min_val=0, max_val=1)
    assert len(c) == 2
    assert c.shape == (2,)


@pytest.mark.unit
def test_qty_math() -> None:
    a = 0.5
    b = Quantity(2)

    assert a + b == 2.5
    assert b + a == 2.5
    assert a - b == -1.5
    assert b - a == 1.5
    assert a * b == 1.0
    assert b * a == 1.0
    np.testing.assert_allclose(a**b, 0.25)
    assert b**a == 2**0.5
    np.testing.assert_allclose(a / b, 0.25)
    assert b / a == 4.0
    assert b % a == 0

    qty = Quantity(3, min_val=0, max_val=5)
    qty.subtract(1.3)
    np.testing.assert_allclose(qty, 1.7)
    qty = Quantity(3, min_val=0, max_val=5)
    qty.add(0.3)
    np.testing.assert_allclose(qty, 3.3)


@pytest.mark.unit
def get_and_set() -> None:
    np.testing.assert_allclose(
        Quantity(0.3, min_val=0, max_val=1).get_opt_value(), [-0.4]
    )
    for val in np.linspace(0, 2):
        a = Quantity(val, min_val=-1, max_val=2)
        opt_val = a.get_opt_value()
        a.set_opt_value(opt_val)
        np.testing.assert_allclose(a, val)
