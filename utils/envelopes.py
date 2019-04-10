"""Library of envelope functions"""

import scipy


def flattop(t, T_up, T_down):
    """
    Flattop gaussian with wixed width of 1 ns, made from erf functions.
    """
    erf = scipy.special.erf
    T2 = max(T_up, T_down)
    T1 = min(T_up, T_down)
    return (1 + erf((t - T1) / 1e-9)) / 2 * (1 + erf((-t + T2) / 1e-9)) / 2
