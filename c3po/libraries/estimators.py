"""Collection of estimator functions."""

import tensorflow as tf
import tensorflow_probability as tfp


def median_dist(exp_values, sim_values, exp_stds):
    """Return the median of the differences."""
    diffs = tf.abs(tf.subtract(exp_values, sim_values))
    return tfp.stats.percentile(diffs, 50.0, interpolation='midpoint')


def rms_dist(exp_values, sim_values, exp_stds):
    """Return the root mean squared of the differences."""
    diffs = tf.abs(tf.subtract(exp_values, tf.transpose(sim_values)))
    return tf.sqrt(tf.reduce_mean(diffs ** 2))


def exp_stds_dist(exp_values, sim_values, exp_stds):
    """Return the mean distance in exp_stds."""
    diffs = tf.abs(tf.subtract(exp_values, sim_values))
    return tf.reduce_mean(diffs / exp_stds)


def rms_exp_stds_dist(exp_values, sim_values, exp_stds):
    """Return the mean distance in exp_stds."""
    diffs = tf.abs(tf.subtract(exp_values, sim_values))
    return tf.sqrt(tf.reduce_mean((diffs / exp_stds)**2))


def std_of_diffs(exp_values, sim_values, exp_stds):
    """Return the mean distance in exp_stds."""
    diffs = tf.abs(tf.subtract(exp_values, sim_values))
    return tf.math.reduce_std(diffs)


def neg_loglkh_binom(exp_values, sim_values, exp_stds):
    """
    Likelihood of the experimental values with binomial distribution.

    Return the likelihood of the experimental values given the simulated
    values, and given a binomial distribution function.
    """
    shots = tf.constant(500., dtype=tf.float64)
    binom = tfp.distributions.Binomial(total_count=shots, probs=sim_values)
    loglkhs = binom.log_prob(exp_values*shots)
    loglkh = tf.reduce_sum(loglkhs)
    # print(sim_values)
    # print(exp_values)
    # print(loglkhs)
    return -loglkh


def neg_loglkh_multinom(exp_values, sim_values, exp_stds):
    """
    Likelihood of the experimental values with multinomial distribution.

    Return the likelihood of the experimental values given the simulated
    values, and given a multinomial distribution function.
    """
    shots = tf.constant(500., dtype=tf.float64)
    binom = tfp.distributions.Multinomial(total_count=shots, probs=sim_values)
    loglkhs = binom.log_prob(exp_values*shots)
    loglkh = tf.reduce_sum(loglkhs)
    # print(sim_values)
    # print(exp_values)
    # print(loglkhs)
    return -loglkh


def neg_loglkh_gauss(exp_values, sim_values, exp_stds):
    """
    Likelihood of the experimental values.

    The distribution is assumed to be binomial (approximated by a gaussian),
    plus an extra fixed gaussian noise distribution (here set at 0.0125)
    """
    std_b = tf.sqrt(sim_values*(1-sim_values))
    mean_b = sim_values
    std_g = 0.0125
    mean_g = 0.
    mean = mean_b + mean_g
    std = tf.sqrt(std_g**2 + std_b**2)
    gauss = tfp.distributions.Normal(mean, std)
    loglkhs = gauss.log_prob(exp_values)
    loglkh = tf.reduce_sum(loglkhs)
    # print('sim')
    # print(sim_values)
    # print('exp')
    # print(exp_values)
    # print('log_likelihood')
    # print(loglkhs)
    # print('\n')
    return -loglkh


def neg_loglkh_mean_gauss_new(exp_values, sim_values, exp_stds):
    """
    Likelihood of the experimental values.
    The distribution is assumed to be binomial (approximated by a gaussian),
    plus an extra fixed gaussian noise distribution (here set at 0.0125)
    """
    std_b = tf.sqrt(sim_values*(1-sim_values))
    mean_b = sim_values
    std_g = 0.
    mean_g = 0.
    mean = mean_b + mean_g
    std = tf.sqrt(std_g**2 + std_b**2)
    gauss = tfp.distributions.Normal(mean, std)
    loglkhs = gauss.log_prob(exp_values)
    loglkhs = loglkhs - gauss.log_prob(mean)
    loglkh = tf.reduce_mean(loglkhs)
    return -loglkh


def neg_loglkh_binom_gauss(exp_values, sim_values, exp_stds):
    """
    Likelihood of the experimental values. CONVOLUTION NOT WORKING.

    To fix this problem we would need to convolve over a large range of values,
    including negative, and then select out the ones of interest.

    https://towardsdatascience.com/
    differentiable-convolution-of-
    probability-distributions-with-tensorflow-79c1dd769b46
    """
    shots = tf.constant(500., dtype=tf.float64)
    binom = tfp.distributions.Binomial(total_count=shots, probs=sim_values)
    gauss = tfp.distributions.Normal(0., 0.0125*shots)
    # dimensions of this input (for NWC format) are
    # [batch, in_width, in_channels]
    # dimensions of the filter are
    # [filter_width, in_channels, out_channels]
    # note the minus sign of x
    lkhs = tf.nn.conv1d(
        tf.reshape(binom.prob(exp_values*shots), (1, -1, 1)),
        tf.reshape(gauss.prob(-exp_values*shots), (-1, 1, 1)),
        stride=1,
        padding='SAME',
        data_format='NWC'
    )
    loglkhs = tf.math.log(lkhs)
    loglkh = tf.reduce_sum(loglkhs)
    # print(sim_values)
    # print(exp_values)
    # print(lkhs)
    # print(loglkhs)
    return -loglkh
