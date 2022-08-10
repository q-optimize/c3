import numpy as np
import pytest

from numpy.testing import assert_array_almost_equal as almost_equal
from c3.signal.gates import Instruction
from c3.libraries.fidelities import (
    unitary_infid,
    average_infid,
    unitary_infid_set,
    state_transfer_infid_set,
)
from c3.libraries.constants import GATES

X = GATES["rxp"]
Y = GATES["ryp"]
Id = GATES["id"]


@pytest.mark.unit
def test_unitary_infid_1() -> None:
    """Testing that a matrix has no error with itself."""
    almost_equal(unitary_infid(X, X, dims=[2]), 0)


@pytest.mark.unit
def test_unitary_infid_2() -> None:
    """Testing that X and Y have maximal error."""
    assert unitary_infid(X, Y, dims=[2]) == 1


@pytest.mark.unit
def test_unitary_infid_3() -> None:
    """Testing that a matrix has no error with itself."""
    actual = np.kron(X, Id)
    almost_equal(unitary_infid(actual, actual, index=[0, 1], dims=[2, 2]), 0)


@pytest.mark.unit
def test_unitary_infid_projection() -> None:
    """Testing only one subspace."""
    actual = np.kron(X, Id)
    almost_equal(unitary_infid(X, actual, index=[0], dims=[2, 2]), 0)


@pytest.mark.unit
def test_unitary_infid_projection_2() -> None:
    """Testing another subspace."""
    actual = np.kron(Id, X)
    almost_equal(unitary_infid(X, actual, index=[1], dims=[2, 2]), 0)


@pytest.mark.unit
def test_unitary_infid_projection_3() -> None:
    """Testing higher levels."""
    actual = np.array(
        [[0 + 0j, 1, 0], [1, 0, 0], [0, 0, 0]],
    )
    almost_equal(unitary_infid(ideal=X, actual=actual, index=[0], dims=[3]), 0)


@pytest.mark.unit
def test_unitary_infid_projection_4() -> None:
    """Testing higher levels."""
    actual = np.array(
        [[0 + 0j, 1, 0], [1, 0, 0], [0, 0, 34345j]],
    )
    almost_equal(unitary_infid(ideal=X, actual=actual, index=[0], dims=[3]), 0)


@pytest.mark.unit
def test_unitary_infid_projection_5() -> None:
    """Testing higher levels and subspaces."""
    actual = np.kron(
        np.array(
            [[0 + 0j, 1, 0], [1, 0, 0], [0, 0, 34345j]],
        ),
        Id,
    )
    almost_equal(unitary_infid(ideal=X, actual=actual, index=[0], dims=[3, 2]), 0)


@pytest.mark.unit
def test_average_infid_1() -> None:
    """Testing that a matrix has no error with itself."""
    almost_equal(average_infid(X, X), 0)


@pytest.mark.unit
def test_average_infid_2() -> None:
    """Testing that X and Y have maximal error."""
    almost_equal(average_infid(X, Y), np.array(2.0 / 3))


@pytest.mark.unit
def test_average_infid_projection() -> None:
    """Testing only one subspace."""
    actual = np.kron(X, Id)
    almost_equal(average_infid(X, actual, index=[0], dims=[2, 2]), 0)


@pytest.mark.unit
def test_average_infid_projection_2() -> None:
    """Testing another subspace."""
    actual = np.kron(Id, X)
    almost_equal(average_infid(X, actual, index=[1], dims=[2, 2]), 0)


@pytest.mark.unit
def test_average_infid_projection_3() -> None:
    """Testing higher levels."""
    actual = np.array(
        [[0 + 0j, 1, 0], [1, 0, 0], [0, 0, 0]],
    )
    almost_equal(average_infid(ideal=X, actual=actual, index=[0], dims=[3]), 0)


@pytest.mark.unit
def test_average_infid_projection_4() -> None:
    """Testing higher levels."""
    actual = np.array(
        [[0 + 0j, 1, 0], [1, 0, 0], [0, 0, 34345j]],
    )
    almost_equal(average_infid(ideal=X, actual=actual, index=[0], dims=[3]), 0)


@pytest.mark.unit
def test_average_infid_projection_5() -> None:
    """Testing higher levels and subspaces."""
    actual = np.kron(
        np.array(
            [[0 + 0j, 1, 0], [1, 0, 0], [0, 0, 34345j]],
        ),
        Id,
    )
    almost_equal(average_infid(ideal=X, actual=actual, index=[0], dims=[3, 2]), 0)


@pytest.mark.unit
def test_set_unitary() -> None:
    propagators = {"rxp": X, "ryp": Y}
    instructions = {"rxp": Instruction("rxp"), "ryp": Instruction("ryp")}
    goal = unitary_infid_set(
        propagators=propagators,
        instructions=instructions,
        index=[0],
        dims=[2],
        n_eval=136,
    )
    almost_equal(goal, 0)


@pytest.mark.unit
def test_set_states() -> None:
    propagators = {"rxp": X, "ryp": Y}
    instructions = {"rxp": Instruction("rxp"), "ryp": Instruction("ryp")}
    psi_0 = np.array([[1], [0]])
    goal = state_transfer_infid_set(
        propagators=propagators,
        instructions=instructions,
        index=[0],
        dims=[2],
        psi_0=psi_0,
        n_eval=136,
    )
    almost_equal(goal, 0)
