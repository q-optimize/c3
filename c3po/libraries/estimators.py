"""Collection of estimator functions."""

import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp


estimators = dict()
def estimator_reg_deco(func):
    """
    Decorator for making registry of functions
    """
    estimators[str(func.__name__)] = func
    return func

@estimator_reg_deco
def mean_dist(exp_values, sim_values, exp_stds, shots):
    """Return the root mean squared of the differences."""
    diffs = tf.abs(tf.subtract(exp_values, sim_values))
    return tf.reduce_mean(diffs)


@estimator_reg_deco
def median_dist(exp_values, sim_values, exp_stds, shots):
    """Return the median of the differences."""
    diffs = tf.abs(tf.subtract(exp_values, sim_values))
    return tfp.stats.percentile(diffs, 50.0, interpolation='midpoint')


@estimator_reg_deco
def rms_dist(exp_values, sim_values, exp_stds, shots):
    """Return the root mean squared of the differences."""
    diffs = tf.abs(tf.subtract(exp_values, sim_values))
    return tf.sqrt(tf.reduce_mean(diffs ** 2))


@estimator_reg_deco
def mean_sim_stds_dist(exp_values, sim_values, exp_stds, shots):
    """Return the mean of the distance in number of exp_stds away."""
    sim_std = tf.sqrt(sim_values*(1-sim_values)/shots)
    diffs = tf.abs(tf.subtract(exp_values, sim_values))
    return tf.reduce_mean(diffs / sim_std)


@estimator_reg_deco
def rms_sim_stds_dist(exp_values, sim_values, exp_stds, shots):
    """Return the root mean squared of the differences measured in exp_stds."""
    sim_std = tf.sqrt(sim_values*(1-sim_values)/shots)
    diffs = tf.abs(tf.subtract(exp_values, sim_values))
    return tf.sqrt(tf.reduce_mean((diffs / sim_std)**2))


@estimator_reg_deco
def mean_exp_stds_dist(exp_values, sim_values, exp_stds, shots):
    """Return the mean of the distance in number of exp_stds away."""
    diffs = tf.abs(tf.subtract(exp_values, sim_values))
    return tf.reduce_mean(diffs / exp_stds)


@estimator_reg_deco
def rms_exp_stds_dist(exp_values, sim_values, exp_stds, shots):
    """Return the root mean squared of the differences measured in exp_stds."""
    diffs = tf.abs(tf.subtract(exp_values, sim_values))
    return tf.sqrt(tf.reduce_mean((diffs / exp_stds)**2))


@estimator_reg_deco
def std_of_diffs(exp_values, sim_values, exp_stds, shots):
    """Return the std of the distances."""
    diffs = tf.abs(tf.subtract(exp_values, sim_values))
    return tf.math.reduce_std(diffs)


@estimator_reg_deco
def neg_loglkh_binom(exp_values, sim_values, exp_stds, shots):
    """
    Average likelihood of the experimental values with binomial distribution.

    Return the likelihood of the experimental values given the simulated
    values, and given a binomial distribution function.
    """
    binom = tfp.distributions.Binomial(total_count=shots, probs=sim_values)
    loglkhs = binom.log_prob(exp_values*shots)
    loglkh = tf.reduce_mean(loglkhs)
    return -loglkh


@estimator_reg_deco
def neg_loglkh_binom_norm(exp_values, sim_values, exp_stds, shots):
    """
    Average likelihood of the exp values with normalised binomial distribution.

    Return the likelihood of the experimental values given the simulated
    values, and given a binomial distribution function that is normalised to
    give probability 1 at the top of the distribution.
    """

    binom = tfp.distributions.Binomial(total_count=shots, probs=sim_values)
    loglkhs = binom.log_prob(exp_values*shots) - \
        binom.log_prob(sim_values*shots)
    loglkh = tf.reduce_mean(loglkhs)
    return -loglkh


@estimator_reg_deco
def neg_loglkh_gauss(exp_values, sim_values, exp_stds, shots):
    """
    Likelihood of the experimental values.

    The distribution is assumed to be binomial (approximated by a gaussian).
    """
    std = tf.sqrt(sim_values*(1-sim_values)/shots)
    mean = sim_values
    gauss = tfp.distributions.Normal(mean, std)
    loglkhs = gauss.log_prob(exp_values)
    loglkh = tf.reduce_mean(loglkhs)
    return -loglkh


@estimator_reg_deco
def neg_loglkh_gauss_norm(exp_values, sim_values, exp_stds, shots):
    """
    Likelihood of the experimental values.

    The distribution is assumed to be binomial (approximated by a gaussian)
    that is normalised to give probability 1 at the top of the distribution.
    """
    std = tf.sqrt(sim_values*(1-sim_values)/shots)
    mean = sim_values
    gauss = tfp.distributions.Normal(mean, std)
    loglkhs = gauss.log_prob(exp_values) - gauss.log_prob(mean)
    loglkh = tf.reduce_mean(loglkhs)
    return -loglkh


@estimator_reg_deco
def neg_loglkh_gauss_norm_sum(exp_values, sim_values, exp_stds, shots):
    """
    Likelihood of the experimental values.

    The distribution is assumed to be binomial (approximated by a gaussian)
    that is normalised to give probability 1 at the top of the distribution.
    """
    std = tf.sqrt(sim_values*(1-sim_values)/shots)
    mean = sim_values
    gauss = tfp.distributions.Normal(mean, std)
    loglkhs = gauss.log_prob(exp_values) - gauss.log_prob(mean)
    loglkh = tf.reduce_sum(loglkhs)

    return -loglkh


@estimator_reg_deco
def g_LL_prime(exp_values, sim_values, exp_stds, shots):
    """
    Likelihood of the experimental values.

    The distribution is assumed to be binomial (approximated by a gaussian)
    that is normalised to give probability 1 at the top of the distribution.
    """
    std = tf.sqrt(sim_values*(1-sim_values)/shots)
    mean = sim_values
    gauss = tfp.distributions.Normal(mean, std)

    prefac = tf.math.sqrt(2 * tf.constant(np.pi) * tf.constant(np.e))
    rescale = tf.math.multiply(tf.cast(prefac, dtype=tf.float64), std)
    #print(rescale)

    loglkhs = tf.math.add(gauss.log_prob(exp_values), tf.math.log(rescale))
    loglkh = tf.reduce_sum(loglkhs)

    K = len(sim_values)
    print("K: " + str(K))
    loglkh = - (1 / K) * loglkh
    return tf.sqrt(loglkh)


@estimator_reg_deco
def neg_loglkh_multinom(exp_values, sim_values, exp_stds, shots):
    """
    Average likelihood of the experimental values with multinomial distribution.

    Return the likelihood of the experimental values given the simulated
    values, and given a multinomial distribution function.
    """
    multi = tfp.distributions.Multinomial(
        total_count=tf.reshape(shots, [shots.shape[0]]),
        probs=sim_values)
    loglkhs = multi.log_prob(exp_values*shots)
    loglkh = tf.reduce_mean(loglkhs)
    return -loglkh


@estimator_reg_deco
def neg_loglkh_multinom_norm(exp_values, sim_values, exp_stds, shots):
    """
    Average likelihood of the experimental values with multinomial distribution.

    Return the likelihood of the experimental values given the simulated
    values, and given a multinomial distribution function that is normalised to
    give probability 1 at the top of the distribution.
    """
    multi = tfp.distributions.Multinomial(
        total_count=tf.reshape(shots, [shots.shape[0]]),
        probs=sim_values)
    loglkhs = multi.log_prob(exp_values*shots) - \
        multi.log_prob(sim_values*shots)
    loglkh = tf.reduce_mean(loglkhs)
    return -loglkh
