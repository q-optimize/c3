"""Library of envelope functions"""

import numpy as np
from scipy.special import erf


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
    T2 = max(T_up, T_down)
    T1 = min(T_up, T_down)
    return (1 + erf((t - T1) / 1e-9)) / 2 * \
           (1 + erf((-t + T2) / 1e-9)) / 2


def gaussian(t, T_final, sigma):
    """
    Normalized gaussian
    """
    sigma = T_final / 6
    gauss = np.exp(-(t - T_final / 2) ** 2 / (2 * sigma ** 2)) - \
        np.exp(-T_final ** 2 / (8 * sigma ** 2))
    norm = np.sqrt(2 * np.pi * sigma ** 2) \
        * erf(T_final / (np.sqrt(8) * sigma)) \
        - T_final * np.exp(-T_final ** 2 / (8 * sigma ** 2))
    # the erf factor takes care of cutoffs at the tails of the gaussian
    return gauss / norm


def gaussian_der(t, T_final, sigma):
    """
    Derivative of the normalized gaussian (ifself not normalized)
    """
    sigma = T_final / 6
    gauss_der = np.exp(-(t - T_final / 2) ** 2 / (2 * sigma ** 2)) * \
        (t - T_final / 2) / sigma ** 2
    norm = np.sqrt(2 * np.pi * sigma ** 2) \
        * erf(T_final / (np.sqrt(8) * sigma)) \
        - T_final * np.exp(-T_final ** 2 / (8 * sigma ** 2))
    return gauss_der / norm
