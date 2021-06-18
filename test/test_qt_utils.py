"""
Test Module for qt_utils
"""
import numpy as np
import pytest
from c3.utils.qt_utils import (
    pauli_basis,
    basis,
    xy_basis,
    get_basis_matrices,
    rotation,
    np_kron_n,
    hilbert_space_kron,
    kron_ids,
    projector,
    pad_matrix,
    perfect_parametric_gate,
    T1_sequence,
    ramsey_sequence,
    ramsey_echo_sequence,
)
from numpy.testing import assert_array_almost_equal as almost_equal
from c3.libraries.constants import GATES


@pytest.fixture()
def test_dimensions() -> list:
    """Functions that allow arbitrary numbers of dimensions will be tested for all dimensions in this list."""
    return [3, 5, 10, 50]


@pytest.mark.unit
def test_pauli_basis() -> None:
    """Testing dimensions of Pauli basis"""
    dims = np.random.randint(2, 5, np.random.randint(1, 5))
    result = pauli_basis(dims)
    assert result.shape[0] == result.shape[1]
    assert result.shape[0] == dims.prod() ** 2


@pytest.mark.unit
def test_basis(test_dimensions) -> None:
    """Testing orthonormality of basis vectors."""
    for dim in test_dimensions:
        pairs = [(i, j) for i in range(dim) for j in range(dim)]
        for (i, j) in pairs:
            vi = basis(dim, i)
            vj = basis(dim, j)
            almost_equal(vi.T @ vj, 1 if i == j else 0)


@pytest.mark.unit
def test_xy_basis(test_dimensions) -> None:
    """Testing properties of basis vectors."""
    names = ["x", "y", "z"]

    for dim in test_dimensions:
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
def test_basis_matrices(test_dimensions) -> None:
    """Testing orthogonality and normalisation of basis matrices."""
    for dim in test_dimensions[:3]:
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
    """Testing trace and determinant of general rotation matrix"""
    phase = 2 * np.pi * np.random.random()
    xyz = np.random.random(3)
    xyz /= np.linalg.norm(xyz)
    matrix = rotation(phase, xyz)

    almost_equal(np.trace(matrix), 2 * np.cos(0.5 * phase))
    almost_equal(np.linalg.det(matrix), 1)


@pytest.mark.unit
def test_np_kron_n(test_dimensions) -> None:
    """Testing properties of Kronecker product"""
    for dim in test_dimensions:
        (A, B, C, D) = [np.random.rand(dim, dim) for _ in range(4)]

        # associativity and mixed product
        almost_equal(np_kron_n([A, B + C]), np_kron_n([A, B]) + np_kron_n([A, C]))
        almost_equal(np_kron_n([A, B]) * np_kron_n([C, D]), np_kron_n([A * C, B * D]))
        # trace and determinant
        almost_equal(np.trace(np_kron_n([A, B])), np.trace(A) * np.trace(B))
        almost_equal(
            np.linalg.det(np_kron_n([A, B])),
            np.linalg.det(A) ** dim * np.linalg.det(B) ** dim,
        )


@pytest.mark.unit
def test_hilbert_space_kron() -> None:
    """Testing dimensions of Kronecker product"""
    dims = np.random.randint(1, 10, 5)
    index = np.random.randint(0, len(dims))
    op_size = dims[index]

    result = hilbert_space_kron(np.random.rand(op_size, op_size), index, dims)
    assert result.shape[0] == result.shape[1]
    assert result.shape[0] == dims.prod()


@pytest.mark.unit
def test_kron_ids() -> None:
    """Testing Kronecker product with identities"""
    # create Kronecker product for some random dimensions and indices
    dims = np.random.randint(2, 10, 3)
    indices = np.where(np.random.rand(len(dims)) > 0.5)[0]
    remaining_indices = np.delete(np.arange(len(dims)), indices)
    matrices = [np.random.rand(dim, dim) for dim in dims[indices]]
    result = kron_ids(dims, indices, matrices)

    # expected dimensions
    assert result.shape[0] == result.shape[1]
    assert result.shape[0] == dims.prod()

    # trace
    traces = np.array([np.trace(X) for X in matrices])
    almost_equal(np.trace(result), traces.prod() * np.prod(dims[remaining_indices]))


@pytest.mark.unit
def test_projector() -> None:
    """Testing subspace projection matrix"""
    # create projector for some random dimensions and indices
    dims = np.random.randint(2, 10, 5)
    indices = np.where(np.random.rand(len(dims)) > 0.5)[0]
    result = projector(dims, indices)

    # check expected dimensions
    assert result.shape[0] == dims.prod()
    expected_dims = np.array([2] * len(indices) + [1] * (len(dims) - len(indices)))
    assert result.shape[1] == expected_dims.prod()


@pytest.mark.unit
def test_pad_matrix(test_dimensions) -> None:
    """Testing shape, trace, and determinant of matrices after padding"""
    for dim in test_dimensions:
        M = np.random.rand(dim, dim)
        padding_dim = np.random.randint(1, 10)

        # padding with unity
        padded_ones = pad_matrix(M, padding_dim, "fulluni")
        assert padded_ones.shape[0] == padded_ones.shape[1]
        almost_equal(padded_ones.shape[0], M.shape[0] + padding_dim)
        almost_equal(np.linalg.det(padded_ones), np.linalg.det(M))
        almost_equal(np.trace(padded_ones), np.trace(M) + padding_dim)

        # padding with zeros
        padded_zeros = pad_matrix(M, padding_dim, "wzeros")
        assert padded_zeros.shape[0] == padded_zeros.shape[1]
        almost_equal(padded_ones.shape[0], M.shape[0] + padding_dim)
        almost_equal(np.linalg.det(padded_zeros), 0)
        almost_equal(np.trace(padded_zeros), np.trace(M))


@pytest.mark.unit
def test_perfect_parametric_gate() -> None:
    """Testing shape and unitarity of the gate matrix"""
    possible_gates = ["X", "Y", "Z", "Id"]
    num_gates = np.random.randint(1, 5)
    gates_str = ":".join(
        np.take(possible_gates, np.random.randint(0, len(possible_gates), num_gates))
    )
    dims = np.random.randint(2, 5, num_gates)
    angle = 2 * np.pi * np.random.rand()
    result = perfect_parametric_gate(gates_str, angle, dims)

    # dimension
    assert result.shape[0] == result.shape[1]
    assert result.shape[0] == dims.prod()

    # unitarity
    almost_equal(result * np.matrix(result).H, np.eye(dims.prod()))


@pytest.mark.unit
def test_sequence_generating_functions() -> None:
    """Testing if the output of all utility functions that generate a sequence of gates contains strings
    from the GATES list"""
    length = np.random.randint(1, 100)
    target = 1
    targetString = "[%d]" % target

    functions = [
        T1_sequence,
        ramsey_sequence,
        ramsey_echo_sequence,
    ]
    for func in functions:
        sequence = func(length, target)
        for gate in sequence:
            assert type(gate) == str
            gateWithoutTarget = gate.replace(targetString, "").lower()
            assert gateWithoutTarget in GATES
