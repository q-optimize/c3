"""Test Module for tf_utils
"""
import tensorflow as tf
from c3.utils.tf_utils import (
    tf_convolve,
    tf_convolve_legacy,
    tf_unitary_overlap,
    tf_measure_operator,
    tf_super,
    Id_like,
    tf_spre,
    tf_spost,
    tf_kron,
)
import pytest
import pickle
from typing import Tuple, List
import numpy as np

with open("test/test_tf_utils.pickle", "rb") as filename:
    data = pickle.load(filename)


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


@pytest.mark.tensorflow
@pytest.mark.unit
def test_convolution_legacy() -> None:
    sigA = data["convolution"]["sigA"]
    sigB = data["convolution"]["sigB"]
    out = tf_convolve_legacy(sigA, sigB)
    np.testing.assert_almost_equal(out, data["convolution"]["out"], decimal=8)


@pytest.mark.tensorflow
@pytest.mark.unit
def test_convolution() -> None:
    sigA = data["convolution"]["sigA"]
    sigB = data["convolution"]["sigB"]
    out = tf_convolve(sigA, sigB)
    np.testing.assert_almost_equal(out, data["convolution"]["out_new"], decimal=8)


@pytest.mark.tensorflow
@pytest.mark.unit
def test_measure_operator():
    for i in range(2, 10):
        M = np.random.rand(i, i)
        rho = np.random.rand(i, i)
        out = tf_measure_operator(M, rho)
        desired = np.trace(M @ rho)
        np.testing.assert_almost_equal(actual=out, desired=desired)


# TODO extend for batched dimensions
@pytest.mark.tensorflow
@pytest.mark.unit
def test_tf_super():
    for el in data["tf_super"]:
        np.testing.assert_allclose(actual=tf_super(el["in"]), desired=el["desired"])


@pytest.mark.tensorflow
@pytest.mark.unit
def test_Id_like():
    for el in data["Id_like"]:
        np.testing.assert_allclose(actual=Id_like(el["in"]), desired=el["desired"])


@pytest.mark.tensorflow
@pytest.mark.unit
def test_tf_spre():
    for el in data["tf_spre"]:
        np.testing.assert_allclose(actual=tf_spre(el["in"]), desired=el["desired"])


@pytest.mark.tensorflow
@pytest.mark.unit
def test_tf_spost():
    for el in data["tf_spost"]:
        np.testing.assert_allclose(actual=tf_spost(el["in"]), desired=el["desired"])


@pytest.mark.tensorflow
@pytest.mark.unit
def test_tf_kron():
    for el in data["tf_kron"]:
        np.testing.assert_allclose(actual=tf_kron(*el["in"]), desired=el["desired"])
