"""Collection of estimator functions."""

import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from c3po.tf_utils import tf_abs


def median_dist(exp_values, sim_values, stds):
    """Return the median of the differences."""
    diffs = tf.abs(tf.subtract(exp_values, sim_values))
    return tfp.stats.percentile(diffs, 50.0, interpolation='midpoint')


def rms_dist(exp_values, sim_values, stds):
    """Return the root mean squared of the differences."""
    diffs = tf.abs(tf.subtract(exp_values, sim_values))
    return tf.sqrt(tf.reduce_mean(diffs ** 2))


def stds_dist(exp_values, sim_values, stds):
    """Return the mean distance in stds."""
    diffs = tf.abs(tf.subtract(exp_values, sim_values))
    return tf.reduce_mean(diffs / stds)


def neg_loglkh_binom(exp_values, sim_values, stds):
    """
    Likelihood of the experimental values with binomial distribution.

    Return the likelihood of the experimental values given the simulated
    values, and given a binomial distribution function.
    """
    print(sim_values)
    print(exp_values)
    binom = tfp.distributions.Binomial(total_count=500, probs=sim_values)
    loglkhs = binom.log_prob(exp_values)
    print(loglkhs)
    loglkh = tf.reduce_sum(loglkhs)
    return -loglkh


def neg_loglkh_binom_gauss(exp_values, sim_values, stds):
    """
    Likelihood of the experimental values.

    https://towardsdatascience.com/
    differentiable-convolution-of-probability-distributions-with-tensorflow-79c1dd769b46
    """
    print(sim_values)
    print(exp_values)
    shots = 500
    binom = tfp.distributions.Binomial(total_count=shots, probs=sim_values)
    gauss = tfp.distributions.Normal(0., 0.0125*shots)
    # dimensions of this input (for NWC format) are
    # [batch, in_width, in_channels]
    # dimensions of the filter are
    # [filter_width, in_channels, out_channels]
    # note the minus sign of x
    lkhs = tf.nn.conv1d(
        tf.reshape(binom.prob(exp_values), (1, -1, 1)),
        tf.reshape(gauss.prob(-exp_values), (-1, 1, 1)),
        stride=1,
        padding='SAME',
        data_format='NWC'
    )
    print(lkhs)
    loglkhs = tf.math.log(lkhs)
    print(loglkhs)
    loglkh = tf.reduce_sum(loglkhs)
    return -loglkh
