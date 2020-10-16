"""
Library of envelope functions.

All functions assume the input of a time vector.
"""

import numpy as np
import tensorflow as tf
from c3.c3objs import Quantity as Qty


def no_drive(t, params):
    """Do nothing."""
    return tf.zeros_like(t, dtype=tf.float64)


def pwc(t, params):
    """Piecewise constant pulse."""
    # TODO make pwc return actual values like other envelopes
    return params


def fourier_sin(t, params):
    """Fourier basis of the pulse constant pulse (sin).

    Parameters
    ----------
    params : dict
        amps : list
            Weights of the fourier components
        freqs : list
            Frequencies of the fourier components

    """
    amps = tf.reshape(
                tf.cast(params['amps'].get_value(), dtype=tf.float64),
                [params['amps'].shape[0], 1]
           )
    freqs = tf.reshape(
                tf.cast(params['freqs'].get_value(), dtype=tf.float64),
                [params['freqs'].shape[0], 1]
           )
    t = tf.reshape(
                tf.cast(t, dtype=tf.float64),
                [1, t.shape[0]]
           )
    return tf.reduce_sum(amps * tf.sin(freqs * t), 0)


def fourier_cos(t, params):
    """Fourier basis of the pulse constant pulse (cos).

    Parameters
    ----------
    params : dict
        amps : list
            Weights of the fourier components
        freqs : list
            Frequencies of the fourier components

    """
    amps = tf.reshape(
                tf.cast(params['amps'].get_value(), dtype=tf.float64),
                [params['amps'].shape[0], 1]
           )
    freqs = tf.reshape(
                tf.cast(params['freqs'].get_value(), dtype=tf.float64),
                [params['freqs'].shape[0], 1]
           )
    t = tf.reshape(
                tf.cast(t, dtype=tf.float64),
                [1, t.shape[0]]
           )
    return tf.reduce_sum(amps * tf.cos(freqs * t), 0)


def rect(t, params):
    """Rectangular pulse. Returns 1 at every time step."""
    return tf.ones_like(t, dtype=tf.float64)


def flattop_risefall(t, params):
    """Flattop gaussian with width of length risefall, modelled by error functions.

    Parameters
    ----------
    params : dict
        t_final : float
            Total length of pulse.
        risefall : float
            Length of the ramps. Position of ramps is so that the pulse starts
            with the start of the ramp-up and ends at the end of the ramp-down

    """
    risefall = tf.cast(params['risefall'].get_value(), dtype=tf.float64)
    t_final = tf.cast(params['t_final'].get_value(), dtype=tf.float64)
    t_up = risefall
    t_down = t_final - risefall
    return (1 + tf.math.erf((t - t_up) / risefall)) / 2 * \
           (1 + tf.math.erf((-t + t_down) / risefall)) / 2


def flattop(t, params):
    """Flattop gaussian with width of length risefall, modelled by error functions.

    Parameters
    ----------
    params : dict
        t_up : float
            Center of the ramp up.
        t_down : float
            Center of the ramp down.
        risefall : float
            Length of the ramps.

    """
    t_up = tf.cast(params['t_up'].get_value(), dtype=tf.float64)
    t_down = tf.cast(params['t_down'].get_value(), dtype=tf.float64)
    risefall = tf.cast(params['risefall'].get_value(), dtype=tf.float64)
    return (1 + tf.math.erf((t - t_up) / risefall)) / 2 * \
           (1 + tf.math.erf((-t + t_down) / risefall)) / 2


def gaussian_sigma(t, params):
    """
    Normalized gaussian. Total area is 1, maximum is determined accordingly.

    Parameters
    ----------
    params : dict
        t_final : float
            Total length of the Gaussian.
        sigma: float
            Width of the Gaussian.

    """
    t_final = tf.cast(params['t_final'].get_value(), dtype=tf.float64)
    sigma = tf.cast(params['sigma'].get_value(), dtype=tf.float64)
    gauss = tf.exp(-(t - t_final / 2) ** 2 / (2 * sigma ** 2))
    norm = (tf.sqrt(2 * np.pi * sigma ** 2)
            * tf.math.erf(t_final / (np.sqrt(8) * sigma))
            - t_final * tf.exp(-t_final ** 2 / (8 * sigma ** 2)))
    offset = tf.exp(-t_final ** 2 / (8 * sigma ** 2))
    return (gauss - offset) / norm


def gaussian(t, params):
    """
    Normalized gaussian with fixed time/sigma ratio.

    Parameters
    ----------
    params : dict
        t_final : float
            Total length of the Gaussian.
    """
    DeprecationWarning("Using standard width. Better use gaussian_sigma.")
    params['sigma'] = Qty(
        value=params['t_final'].get_value()/6,
        min=params['t_final'].get_value()/8,
        max=params['t_final'].get_value()/4,
        unit=params['t_final'].unit
    )
    return gaussian_sigma(t, params)


def gaussian_nonorm(t, params):
    """
    Non-normalized gaussian. Maximum value is 1, area is given by length.

    Parameters
    ----------
    params : dict
        t_final : float
            Total length of the Gaussian.
        sigma: float
            Width of the Gaussian.

    """
    # TODO Add zeroes for t>t_final
    t_final = tf.cast(params['t_final'].get_value(), dtype=tf.float64)
    sigma = params['sigma'].get_value()
    gauss = tf.exp(-(t - t_final / 2) ** 2 / (2 * sigma ** 2))
    return gauss


def gaussian_der_nonorm(t, params):
    """Derivative of the normalized gaussian (ifself not normalized)."""
    t_final = tf.cast(params['t_final'].get_value(), dtype=tf.float64)
    sigma = tf.cast(params['sigma'].get_value(), dtype=tf.float64)
    gauss_der = tf.exp(-(t - t_final / 2) ** 2 / (2 * sigma ** 2)) * \
        (t - t_final / 2) / sigma ** 2
    return gauss_der


def gaussian_der(t, params):
    """Derivative of the normalized gaussian (ifself not normalized)."""
    t_final = tf.cast(params['t_final'].get_value(), dtype=tf.float64)
    sigma = tf.cast(params['sigma'].get_value(), dtype=tf.float64)
    gauss_der = tf.exp(-(t - t_final / 2) ** 2 / (2 * sigma ** 2)) * \
        (t - t_final / 2) / sigma ** 2
    norm = tf.sqrt(2 * np.pi * sigma ** 2) \
        * tf.math.erf(t_final / (tf.sqrt(8) * sigma)) \
        - t_final * tf.exp(-t_final ** 2 / (8 * sigma ** 2))
    return gauss_der / norm


def drag_sigma(t, params):
    """Second order gaussian."""
    t_final = tf.cast(params['t_final'].get_value(), dtype=tf.float64)
    sigma = tf.cast(params['sigma'].get_value(), dtype=tf.float64)
    drag = tf.exp(-(t - t_final / 2) ** 2 / (2 * sigma ** 2))
    norm = (tf.sqrt(2 * np.pi * sigma ** 2)
            * tf.math.erf(t_final / (np.sqrt(8) * sigma))
            - t_final * tf.exp(-t_final ** 2 / (8 * sigma ** 2)))
    offset = tf.exp(-t_final ** 2 / (8 * sigma ** 2))
    return (drag - offset) ** 2 / norm


def drag(t, params):
    """Second order gaussian with fixed time/sigma ratio."""
    DeprecationWarning("Using standard width. Better use drag_sigma.")
    params['sigma'] = Qty(
        value=params['t_final'].get_value()/4,
        min=params['t_final'].get_value()/8,
        max=params['t_final'].get_value()/2,
        unit=params['t_final'].unit
    )
    return drag_sigma(t, params)


def drag_der(t, params):
    """Derivative of second order gaussian."""
    t_final = tf.cast(params['t_final'].get_value(), dtype=tf.float64)
    sigma = tf.cast(params['sigma'].get_value(), dtype=tf.float64)
    norm = (tf.sqrt(2 * np.pi * sigma ** 2)
            * tf.math.erf(t_final / (np.sqrt(8) * sigma))
            - t_final * tf.exp(-t_final ** 2 / (8 * sigma ** 2)))
    offset = tf.exp(-t_final ** 2 / (8 * sigma ** 2))
    der = - 2 * (tf.exp(-(t - t_final / 2) ** 2 / (2 * sigma ** 2)) - offset) \
        * (np.exp(-(t - t_final / 2) ** 2 / (2 * sigma ** 2))) \
        * (t - t_final / 2) / sigma ** 2 / norm
    return der


def flattop_variant(t, params):
    """
    Flattop variant.
    """
    t_up = params['t_up']
    t_down = params['t_down']
    ramp = params['ramp']
    value = np.ones(len(t))
    if ramp > (t_down-t_up)/2:
        ramp = (t_down-t_up)/2
    sigma = np.sqrt(2)*ramp*0.2
    for i, e in enumerate(t):
        if t_up <= e <= t_up+ramp:
            value[i] = np.exp(-(e-t_up-ramp)**2/(2*sigma**2))
        elif t_up+ramp < e < t_down-ramp:
            value[i] = 1
        elif t_down >= e >= t_down-ramp:
            value[i] = np.exp(-(e-t_down+ramp)**2/(2*sigma**2))
        else:
            value[i] = 0
    return value
