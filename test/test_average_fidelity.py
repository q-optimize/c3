import tensorflow as tf
import numpy as np
from c3.utils.tf_utils import (
    tf_choi_to_chi, super_to_choi, tf_super, tf_abs, tf_project_to_comp
)
from numpy.testing import assert_array_almost_equal as almost_equal

dims = [3, 3]
lvls = dims
U_actual = np.array(
    [[1, 0, 0, -1.j, 0, 0, 0, 0, 0],
     [0, 1, 0, 0, -1.j, 0, 0, 0, 0],
     [0, 0, 0, 0, 0, 0, 0, 0, 0],
     [-1.j, 0, 0, 1, 0, 0, 0, 0, 0],
     [0, -1.j, 0, 0, 1, 0, 0, 0, 0],
     [0, 0, 0, 0, 0, 0, 0, 0, 0],
     [0, 0, 0, 0, 0, 0, 0, 0, 0],
     [0, 0, 0, 0, 0, 0, 0, 45, 0],
     [0, 0, 0, 0, 0, 0, 0, 0, 0]]
) / np.sqrt(2)
U_ideal = np.array(
    [[1, 0, -1.j, 0],
     [0, 1, 0, -1.j],
     [-1.j, 0, 1, 0],
     [0, -1.j, 0, 1]]) / np.sqrt(2)

d = 4

Lambda = tf.matmul(
    tf.linalg.adjoint(tf_project_to_comp(U_actual, lvls, to_super=False)), U_ideal
)
err = tf_super(Lambda)
choi = super_to_choi(err)
chi = tf_choi_to_chi(choi, dims=lvls)
fid = tf_abs((chi[0, 0] / d + 1) / (d + 1))


def test_error_channel():
    almost_equal(Lambda, np.eye(4))


def test_fid():
    almost_equal(fid, 1)
