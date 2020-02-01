"""Collection of estimator functions."""

import tensorflow as tf
import tensorflow_probability as tfp
from c3po.tf_utils import tf_abs


def median_dist(exp_values, sim_values, stds):
    """Return the median of the differences."""
    diffs = tf_abs(exp_values-sim_values)
    return tfp.stats.percentile(diffs, 50.0, interpolation='midpoint')


def rms_dist(exp_values, sim_values, stds):
    """Return the root mean squared of the differences."""
    diffs = tf_abs(exp_values-sim_values)
    return tf.sqrt(tf.reduce_mean(tf.stack(diffs) ** 2))


def stds_dist(exp_values, sim_values, stds):
    """Return the mean distance in stds."""
    diffs = tf_abs(exp_values-sim_values)
    return tf.reduce_mean(diffs / stds)


def neg_lkh_binom(exp_values, sim_values, stds):
    """
    Likelihood of the experimental values with binomial distribution.

    Return the likelihood of the experimental values given the simulated
    values, and given a binomial distribution function.
    """
    binom = tfp.distributions.Binomial(total_count=400, probs=sim_values)
    lkhs = binom.prob(exp_values)
    lhk = tf.exp(tf.reduce_sum(tf.log(lkhs)))
    return -lhk
