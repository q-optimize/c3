"""Test Module for tf_utils
"""
import tensorflow as tf
from c3.utils.tf_utils import tf_unitary_overlap
import pytest
from typing import Tuple, List


@pytest.mark.tensorflow
@pytest.mark.unit
@pytest.mark.parametrize(
    "args",
    [
        ([[0.0, 1.0], [1.0, 0.0]], [[0.0, 1.0], [1.0, 0.0]], [0.999]),
        ([[0.0, 1.0], [1.0, 0.0]], [[0.0, 1.001], [1.001, 0.0]], [1.001]),
    ],
)
def test_unitary_overlap(args: Tuple[List[int], List[int], List[int]]) -> None:
    """test unitary overlap function from tf_utils

    Parameters
    ----------
    args : Tuple[List[int], List[int], List[int]]
        Matrix A, Matrix B and Expected Overlap
    """
    x, x_noisy, over = args
    pauli_x = tf.constant(x)
    pauli_x_noisy = tf.constant(x_noisy)

    overlap = tf_unitary_overlap(pauli_x, pauli_x_noisy)
    assert overlap.numpy() > over
