"""
Test Module for qt_utils
"""
import numpy as np
import pytest
from c3.utils.qt_utils import basis, xy_basis, get_basis_matrices, rotation, np_kron_n
from numpy.testing import assert_array_almost_equal as almost_equal


@pytest.mark.unit
def test_basis() -> None:
    """Testing orthonormality of basis vectors."""
    for dim in [3, 5, 10, 100]:
        pairs = [(i, j) for i in range(dim) for j in range(dim)]
        for (i, j) in pairs:
            vi = basis(dim, i)
            vj = basis(dim, j)
            almost_equal(vi.T @ vj, 1 if i == j else 0)


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

        # overlap
        pairs = [(a, b) for a in names for b in names if b is not a]
        for (a, b) in pairs:
            va_p = xy_basis(dim, a + "p")
            va_m = xy_basis(dim, a + "m")
            vb_p = xy_basis(dim, b + "p")
            vb_m = xy_basis(dim, b + "m")
            almost_equal(np.linalg.norm(np.vdot(va_p.T, vb_p)), 1.0 / np.sqrt(2))
            almost_equal(np.linalg.norm(np.vdot(va_p.T, vb_m)), 1.0 / np.sqrt(2))
            almost_equal(np.linalg.norm(np.vdot(va_m.T, vb_p)), 1.0 / np.sqrt(2))
            almost_equal(np.linalg.norm(np.vdot(va_m.T, vb_m)), 1.0 / np.sqrt(2))


@pytest.mark.unit
def test_basis_matrices() -> None:
    """Testing properties of basis matrices."""
    for dim in [3, 5, 10]:
        matrices = get_basis_matrices(dim)

        # orthogonality
        pairs = [(a, b) for a in matrices for b in matrices if b is not a]
        for (a, b) in pairs:
            almost_equal(np.linalg.norm(np.multiply(a, b)), 0)

        # normalisation
        for a in matrices:
            almost_equal(np.linalg.norm(np.multiply(a, a)), 1)


@pytest.mark.unit
def test_rotation() -> None:
    """Testing properties of general rotation matrix"""
    phase = 2 * np.pi * np.random.random()
    xyz = np.random.random(3)
    xyz /= np.linalg.norm(xyz)
    matrix = rotation(phase, xyz)

    almost_equal(np.trace(matrix), 2 * np.cos(0.5 * phase))
    almost_equal(np.linalg.det(matrix), 1)


@pytest.mark.unit
def test_np_kron_n() -> None:
    """Testing Kronecker product"""
    for dim in [3, 5, 10]:
        A = np.random.rand(dim, dim)
        B = np.random.rand(dim, dim)
        C = np.random.rand(dim, dim)
        D = np.random.rand(dim, dim)

        # associativity and mixed product
        almost_equal(np_kron_n([A, B + C]), np_kron_n([A, B]) + np_kron_n([A, C]))
        almost_equal(np_kron_n([A, B]) * np_kron_n([C, D]), np_kron_n([A * C, B * D]))
        # trace and determinant
        almost_equal(np.trace(np_kron_n([A, B])), np.trace(A) * np.trace(B))
        almost_equal(
            np.linalg.det(np_kron_n([A, B])),
            np.linalg.det(A) ** dim * np.linalg.det(B) ** dim,
        )
