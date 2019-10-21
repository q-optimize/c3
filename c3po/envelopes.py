"""Library of envelope functions."""

import numpy as np
import tensorflow as tf


def pwc(t, params):
    """Piecewise constant pulse."""
    return params


def flattop_risefall(t, params):
    """Flattop gaussian with width of length risefall."""
    t_up = params['t_up']
    t_down = params['t_down']
    risefall = params['risefall']
    return (1 + tf.math.erf((t - t_down) / risefall)) / 2 * \
           (1 + tf.math.erf((-t + t_up) / risefall)) / 2


def flattop(t, params):
    """Flattop gaussian with fixed width of 1ns."""
    params['risefall'] = 1e-9
    return flattop_risefall(t, params)


def gaussian(t, params):
    """Normalized gaussian."""
    t_final = params['t_final']
    sigma = params['sigma']
    gauss = tf.exp(-(t - t_final / 2) ** 2 / (2 * sigma ** 2))
    norm = (tf.sqrt(2 * np.pi * sigma ** 2)
            * tf.math.erf(t_final / (np.sqrt(8) * sigma))
            - t_final * tf.exp(-t_final ** 2 / (8 * sigma ** 2)))
    offset = tf.exp(-t_final ** 2 / (8 * sigma ** 2))
    return (gauss - offset) / norm


def gaussian_der(t, params):
    """Derivative of the normalized gaussian (ifself not normalized)."""
    t_final = params['t_final']
    sigma = params['sigma']
    gauss_der = tf.exp(-(t - t_final / 2) ** 2 / (2 * sigma ** 2)) * \
        (t - t_final / 2) / sigma ** 2
    norm = tf.sqrt(2 * np.pi * sigma ** 2) \
        * tf.math.erf(t_final / (tf.sqrt(8) * sigma)) \
        - t_final * tf.exp(-t_final ** 2 / (8 * sigma ** 2))
    return gauss_der / norm


def drag(t, params):
    """Second order gaussian."""
    t_final = params['t_final']
    sigma = tf.cast(t_final / 6, tf.float64)
    drag = tf.exp(-(t - t_final / 2) ** 2 / (2 * sigma ** 2))
    norm = (tf.sqrt(2 * np.pi * sigma ** 2)
            * tf.math.erf(t_final / (np.sqrt(8) * sigma))
            - t_final * tf.exp(-t_final ** 2 / (8 * sigma ** 2)))
    offset = tf.exp(-t_final ** 2 / (8 * sigma ** 2))
    return (drag - offset) ** 2 / norm


def drag_der(t, params):
    """Derivative of second order gaussian."""
    t_final = params['t_final']
    sigma = tf.cast(t_final / 6, tf.float64)
    norm = (tf.sqrt(2 * np.pi * sigma ** 2)
            * tf.math.erf(t_final / (np.sqrt(8) * sigma))
            - t_final * tf.exp(-t_final ** 2 / (8 * sigma ** 2)))
    offset = tf.exp(-t_final ** 2 / (8 * sigma ** 2))
    der = - 2 * (tf.exp(-(t - t_final / 2) ** 2 / (2 * sigma ** 2)) - offset) \
        * (np.exp(-(t - t_final / 2) ** 2 / (2 * sigma ** 2))) \
        * (t - t_final / 2) / sigma ** 2 / norm
    return der


def flattop_WMI(t, params):
    """
    Flattop version used at WMI.

    Specification added by Stephan T and Afonso L.
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
