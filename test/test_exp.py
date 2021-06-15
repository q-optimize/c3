"""Testing exponentiantion methods for PWC propagation"""
import tensorflow as tf
import pytest

from numpy.testing import assert_array_almost_equal as almost_equal
from c3.libraries.propagation import tf_expm, tf_expm_dynamic


@pytest.mark.unit
def test_tf_expm(get_exp_problem) -> None:
    """Testing tf_expm with fixed number of terms"""

    rot, res = get_exp_problem
    terms = 100
    almost_equal(tf_expm(1j * rot, terms), res)


@pytest.mark.unit
@pytest.mark.skip(reason="experimental: to be tested")
def test_expm_dynamic(get_exp_problem) -> None:
    """Testing dynamically adjusted exp method"""

    rot, res = get_exp_problem
    almost_equal(tf_expm_dynamic(1j * rot), res)


@pytest.mark.unit
def test_tf_exponentiation(get_exp_problem) -> None:
    """Testing with the TF exponentiation method"""

    rot, res = get_exp_problem
    almost_equal(tf.linalg.expm(1j * rot), res)
