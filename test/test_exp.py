#!/usr/bin/env python
# coding: utf-8

# In[3]:


import numpy as np
import tensorflow as tf
import pytest

from numpy.testing import assert_array_almost_equal as almost_equal
from c3.libraries.constants import Id, X, Y, Z
from c3.libraries.propagation import tf_expm, tf_expm_dynamic

theta = 2 * np.pi * np.random.rand(1)
"""Testing that exponentiation methods are almost equal in the numpy sense.
Check that, given P = a*X+b*Y+c*Z with a, b, c random normalized numbers,
exp(i theta P) = cos(theta)*Id + sin(theta)*P"""
a = np.random.rand(1)
b = np.random.rand(1)
c = np.random.rand(1)
norm = np.sqrt(a ** 2 + b ** 2 + c ** 2)
# Normalized random coefficients
a = a / norm
b = b / norm
c = c / norm

P = a * X + b * Y + c * Z
theta = 2 * np.pi * np.random.rand(1)
rot = theta * P
res = np.cos(theta) * Id + 1j * np.sin(theta) * P


@pytest.mark.unit
def test_tf_expm() -> None:
    terms = 25
    almost_equal(tf_expm(1j * rot, terms), res)


@pytest.mark.unit
@pytest.mark.skip(reason="experimental: to be tested")
def test_expm_dynamic() -> None:
    almost_equal(tf_expm_dynamic(1j * rot), res)


@pytest.mark.unit
def test_tf_exponentiation() -> None:
    almost_equal(tf.linalg.expm(1j * rot), res)
