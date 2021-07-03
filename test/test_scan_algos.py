from typing import List
import numpy as np
from c3.libraries.algorithms import sweep

params = list()

X_INIT = [5.0]
POINTS = 5
BOUNDS = [[0.0, 2.5]]
DESIRED_PARAMS = np.linspace(BOUNDS[0][0], BOUNDS[0][1], POINTS, dtype=float)
INIT_POINT = False


def mock_fun(x: List[float]) -> None:
    params.append(x[0])


def test_sweep() -> None:
    sweep(
        X_INIT,
        fun=mock_fun,
        options={"points": POINTS, "bounds": BOUNDS, "init_point": INIT_POINT},
    )
    np.testing.assert_allclose(params, DESIRED_PARAMS)
