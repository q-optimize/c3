"""Library of envelope functions"""

import numpy as np
import tensorflow as tf
from tensorflow.math import exp, erf, sqrt


def flattop_risefall(t, T_up, T_down, risefall):
    """
    Flattop gaussian with width of length risefall, made from erf functions.
    """
    T2 = max(T_up, T_down)
    T1 = min(T_up, T_down)
    return (1 + erf((t - T1) / risefall)) / 2 * \
           (1 + erf((-t + T2) / risefall)) / 2


def flattop(t, T_up, T_down):
    """
    Flattop gaussian with fixed width of 1ns, made from erf functions.
    """
    T2 = tf.maximum(T_up, T_down)
    T1 = tf.minimum(T_up, T_down)
    return (1 + erf((t - T1) / 1e-9)) / 2 * \
           (1 + erf((-t + T2) / 1e-9)) / 2


def gaussian(t, T_final, sigma):
    """
    Normalized gaussian
    """
    sigma = T_final / 6
    gauss = exp(-(t - T_final / 2) ** 2 / (2 * sigma ** 2)) - \
        exp(-T_final ** 2 / (8 * sigma ** 2))
    norm = sqrt(2 * np.pi * sigma ** 2) \
        * erf(T_final / (sqrt(8) * sigma)) \
        - T_final * exp(-T_final ** 2 / (8 * sigma ** 2))
    # the erf factor takes care of cutoffs at the tails of the gaussian
    return gauss / norm


def gaussian_der(t, T_final, sigma):
    """
    Derivative of the normalized gaussian (ifself not normalized)
    """
    sigma = T_final / 6
    gauss_der = exp(-(t - T_final / 2) ** 2 / (2 * sigma ** 2)) * \
        (t - T_final / 2) / sigma ** 2
    norm = sqrt(2 * np.pi * sigma ** 2) \
        * erf(T_final / (sqrt(8) * sigma)) \
        - T_final * exp(-T_final ** 2 / (8 * sigma ** 2))
    return gauss_der / norm


def flattop_WMI(t, T_up, T_down, ramp=20*10**(-9)):
    """
    Flattop version used at WMI. Specification added by Stephan T and Afonso L.
    """
    T1 = min(T_up, T_down)
    T2 = max(T_up, T_down)
    value = np.ones(len(t))
    if ramp > (T2-T1)/2:
        ramp = (T2-T1)/2
    sigma = sqrt(2)*ramp*0.2
    for i, e in enumerate(t):
        if T1 <= e <= T1+ramp:
            value[i] = exp(-(e-T1-ramp)**2/(2*sigma**2))
        elif T1+ramp < e < T2-ramp:
            value[i] = 1
        elif T2 >= e >= T2-ramp:
            value[i] = exp(-(e-T2+ramp)**2/(2*sigma**2))
        else:
            value[i] = 0
    return value
