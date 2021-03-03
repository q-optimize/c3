"""
Tests for average fidelity and helper functions.
"""

import numpy as np
from numpy.testing import assert_array_almost_equal as almost_equal
import pytest


@pytest.mark.unit
def test_error_channel(get_error_process):
    """
    Check that the given process performs an identity in the computational subspace.
    """
    almost_equal(get_error_process, np.eye(4))


@pytest.mark.unit
def test_fid(get_average_fidelitiy):
    """
    Check that the average fideltiy of an identity is maximal.
    """
    almost_equal(get_average_fidelitiy, 1)
