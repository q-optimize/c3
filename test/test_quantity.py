"""
testing module for Quantity class
"""

from c3.c3objs import Quantity

gate_time = Quantity(
    value=5.3246e-9,
    min=2e-9,
    max=10e-9,
    unit="s",
    symbol=r"t_g"
)

matrix = Quantity(
    value=[[0, 1], [1, 0]],
    min=[[0, 0], [0, 0]],
    max=[[1, 1], [1, 1]],
    unit="",
    symbol=r"M"
)


def test_qty_str() -> None:
    assert str(gate_time) == "5.325 ns "


def test_qty_set() -> None:
    gate_time.set_value(7e-9)
    assert gate_time.get_value() == 7e-9


def test_qty_max() -> None:
    gate_time.set_opt_value(1.0)
    assert gate_time.get_value() == 10e-9


def test_qty_min() -> None:
    gate_time.set_opt_value(-1.0)
    assert gate_time.get_value() == 2e-9


def test_qty_get_opt() -> None:
    gate_time.set_value(6e-9)
    assert gate_time.get_opt_value() < 1e-15


def test_qty_matrix_str() -> None:
    assert str(matrix) == '0.000  1.000  1.000  0.000  '


def test_qty_matrix_set() -> None:
    matrix.set_value(
        [[1.0, 0.0],
         [0.0, 1.0]]
    )
    assert (matrix.numpy() == [[1, 0], [0, 1]]).all()


def test_qty_matrix_set_opt() -> None:
    assert (matrix.get_opt_value() == [1.,  -1.,  -1., 1.]).all()
