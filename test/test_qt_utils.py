"""
Test Module for qt_utils
"""
import numpy as np
import pytest
from c3.utils.qt_utils import basis, xy_basis, get_basis_matrices
from numpy.testing import assert_array_almost_equal as almost_equal


@pytest.mark.unit
def test_basis() -> None:
    """Testing orthonormality of basis vectors."""
    for dim in [3, 5, 10, 100]:
        pairs = [(i, j) for i in range(dim) for j in range(dim)]
        for p in pairs:
            vi = basis(dim, p[0])
            vj = basis(dim, p[1])
            almost_equal(vi.T @ vj, 1 if p[0] == p[1] else 0)


@pytest.mark.unit
def test_xy_basis() -> None:
    """Testing properties of basis vectors."""
    names = ["x", "y", "z"]

    for dim in [3, 5, 10, 100]:
        # orthonormality of +/- vectors
        for i in names:
            vi_p = xy_basis(dim, i + "p")
            vi_m = xy_basis(dim, i + "m")
            almost_equal(np.linalg.norm(vi_p), 1)
            almost_equal(np.linalg.norm(vi_m), 1)
            almost_equal(np.vdot(vi_p.T, vi_m), 0)


@pytest.mark.unit
def test_basis_matrices() -> None:
    """Testing properties of basis matrices."""
    for dim in [3, 5, 10]:
        matrices = get_basis_matrices(dim)

        # orthogonality
        pairs = [(a, b) for a in matrices for b in matrices if b is not a]
        for p in pairs:
            almost_equal(np.linalg.norm(np.multiply(p[0], p[1])), 0)

        # normalisation
        for a in matrices:
            almost_equal(np.linalg.norm(np.multiply(a, a)), 1)
