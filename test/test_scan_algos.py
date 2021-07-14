"""Test module for algorithms that scan a function across a given range.
We use a mock function that appends the parameter passed to it into a global
list. We then cross check this list against expected values.
"""

from typing import List
import numpy as np
from c3.libraries.algorithms import sweep, grid2D

params = list()
params2D = list()

# constants for sweep and grid2D parameters
X_INIT = [5.0]
POINTS = 5
BOUNDS = [[0.0, 2.5]]
BOUNDS_2D = [[0.0, 2.5], [0.0, 2.5]]
DESIRED_PARAMS = np.linspace(BOUNDS[0][0], BOUNDS[0][1], POINTS, dtype=float)
DESIRED_PARAMS_2D = [
    [x, y]
    for x in np.linspace(BOUNDS_2D[0][0], BOUNDS_2D[0][1], POINTS)
    for y in np.linspace(BOUNDS_2D[1][0], BOUNDS_2D[1][1], POINTS)
]
INIT_POINT = False


def mock_fun(x: List[float]) -> None:
    params.append(x[0])


def mock_fun2D(inputs: List[float]) -> None:
    params2D.append(inputs)


def test_sweep() -> None:
    """Test c3.libraries.algorithms.sweep()"""
    sweep(
        X_INIT,
        fun=mock_fun,
        options={"points": POINTS, "bounds": BOUNDS, "init_point": INIT_POINT},
    )
    np.testing.assert_allclose(params, DESIRED_PARAMS)


def test_grid2D() -> None:
    """Test c3.libraries.algorithms.grid2D()"""
    grid2D(X_INIT, fun=mock_fun2D, options={"points": POINTS, "bounds": BOUNDS_2D})
    np.testing.assert_allclose(params2D, DESIRED_PARAMS_2D)
