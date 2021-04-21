"""
Library of envelope functions.

All functions assume the input of a time vector.
"""

import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from c3.c3objs import Quantity as Qty

envelopes = dict()


def env_reg_deco(func):
    """
    Decorator for making registry of functions
    """
    envelopes[str(func.__name__)] = func
    return func


@env_reg_deco
def no_drive(t, params=None):
    """Do nothing."""
    return tf.zeros_like(t, dtype=tf.float64)


@env_reg_deco
def pwc(t, params):
    """Piecewise constant pulse."""
    # TODO make pwc return actual values like other envelopes
    return params


def pwc_shape(t, params):
    t_bin_start = tf.cast(params["t_bin_end"].get_value(), tf.float64)
    t_bin_end = tf.cast(params["t_bin_start"].get_value(), tf.float64)
    # t_final = tf.cast(params["t_final"].get_value(), tf.float64)
    inphase = tf.cast(params["inphase"].get_value(), tf.float64)

    t_interp = t
    shape = tf.reshape(
        tfp.math.interp_regular_1d_grid(
            t_interp,
            t_bin_start,
            t_bin_end,
            inphase,
            fill_value_below=0,
            fill_value_above=0,
        ),
        [len(t)],
    )

    return shape


@env_reg_deco
def pwc_symmetric(t, params):
    """symmetic PWC pulse
    This works only for inphase component"""
    t_bin_start = tf.cast(params["t_bin_end"].get_value(), tf.float64)
    t_bin_end = tf.cast(params["t_bin_start"].get_value(), tf.float64)
    t_final = tf.cast(params["t_final"].get_value(), tf.float64)
    inphase = tf.cast(params["inphase"].get_value(), tf.float64)

    t_interp = tf.where(tf.greater(t, t_final / 2), -t + t_final, t)
    shape = tf.reshape(
        tfp.math.interp_regular_1d_grid(
            t_interp,
            t_bin_start,
            t_bin_end,
            inphase,
            fill_value_below=0,
            fill_value_above=0,
        ),
        [len(t)],
    )
    return shape


@env_reg_deco
def delta_pulse(t, params):
    "Pulse shape which gives an output only at a given time bin"
    t_sig = tf.cast(params["t_sig"].get_value(), tf.float64)
    shape = tf.zeros_like(t)
    for t_s in t_sig:
        shape = tf.where(
            tf.reduce_min((t - t_s - 1e-9) ** 2) == (t - t_s - 1e-9) ** 2,
            np.ones_like(t),
            shape,
        )
    return shape


@env_reg_deco
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
        tf.cast(params["amps"].get_value(), dtype=tf.float64),
        [params["amps"].get_value().shape[0], 1],
    )
    freqs = tf.reshape(
        tf.cast(params["freqs"].get_value(), dtype=tf.float64),
        [params["freqs"].get_value().shape[0], 1],
    )
    phases = tf.reshape(
        tf.cast(params["phases"].get_value(), dtype=tf.float64),
        [params["phases"].get_value().shape[0], 1],
    )
    t = tf.reshape(tf.cast(t, tf.float64), [1, t.shape[0]])
    return tf.reduce_sum(amps * tf.sin(freqs * t + phases), 0)


@env_reg_deco
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
        tf.cast(params["amps"].get_value(), tf.float64), [params["amps"].shape[0], 1]
    )
    freqs = tf.reshape(
        tf.cast(params["freqs"].get_value(), tf.float64), [params["freqs"].shape[0], 1]
    )
    t = tf.reshape(tf.cast(t, tf.float64), [1, t.shape[0]])
    return tf.reduce_sum(amps * tf.cos(freqs * t), 0)


@env_reg_deco
def rect(t, params):
    """Rectangular pulse. Returns 1 at every time step."""
    return tf.ones_like(t, tf.float64)


@env_reg_deco
def trapezoid(t, params):
    """Trapezoidal pulse. Width of linear slope.

    Parameters
    ----------
    params : dict
        t_final : float
            Total length of pulse.
        risefall : float
            Length of the slope
    """
    risefall = tf.cast(params["risefall"].get_value(), tf.float64)
    t_final = tf.cast(params["t_final"].get_value(), tf.float64)

    envelope = tf.ones_like(t, tf.float64)
    envelope = tf.where(
        tf.less_equal(t, risefall * 2.5), t / (risefall * 2.5), envelope
    )
    envelope = tf.where(
        tf.greater_equal(t, t_final - risefall * 2.5),
        (t_final - t) / (risefall * 2.5),
        envelope,
    )
    return envelope


@env_reg_deco
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
    risefall = tf.cast(params["risefall"].get_value(), tf.float64)
    t_final = tf.cast(params["t_final"].get_value(), tf.float64)
    t_up = risefall
    t_down = t_final - risefall
    return (
        (1 + tf.math.erf((t - t_up) / risefall))
        / 2
        * (1 + tf.math.erf((-t + t_down) / risefall))
        / 2
    )


@env_reg_deco
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
    t_up = tf.cast(params["t_up"].get_value(), tf.float64)
    t_down = tf.cast(params["t_down"].get_value(), tf.float64)
    risefall = tf.cast(params["risefall"].get_value(), tf.float64)
    return (
        (1 + tf.math.erf((t - t_up) / (risefall)))
        / 2
        * (1 + tf.math.erf((-t + t_down) / (risefall)))
        / 2
    )


@env_reg_deco
def flattop_cut(t, params):
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
    t_up = tf.cast(params["t_up"].get_value(), dtype=tf.float64)
    t_down = tf.cast(params["t_down"].get_value(), dtype=tf.float64)
    risefall = tf.cast(params["risefall"].get_value(), dtype=tf.float64)
    shape = tf.math.erf((t - t_up) / risefall) * tf.math.erf((-t + t_down) / risefall)
    shape = tf.clip_by_value(shape, 0, 1)
    shape /= tf.reduce_max(shape)
    return shape


@env_reg_deco
def flattop_cut_center(t, params):
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
    t_final = tf.cast(params["t_final"].get_value(), tf.float64)
    width = tf.cast(params["width"].get_value(), tf.float64)
    risefall = tf.cast(params["risefall"].get_value(), tf.float64)
    t_up = t_final / 2 - width / 2
    t_down = t_final / 2 + width / 2
    shape = tf.math.erf((t - t_up) / risefall) * tf.math.erf((-t + t_down) / risefall)
    shape = tf.clip_by_value(shape, 0, 2)
    return shape


@env_reg_deco
def slepian_fourier(t, params):
    """
    ----
    """
    t_final = tf.cast(params["t_final"].get_value(), tf.float64)
    width = tf.cast(params["width"].get_value(), tf.float64)
    fourier_coeffs = tf.cast(params["fourier_coeffs"].get_value(), tf.float64)
    offset = tf.cast(params["offset"].get_value(), tf.float64)
    amp = tf.cast(params["amp"].get_value(), tf.float64)
    if "risefall" in params:
        plateau = width - params["risefall"].get_value() * 2
        x = tf.identity(t)
        x = tf.where(t > (t_final + plateau) / 2, t - plateau / 2, x)
        x = tf.where(t < (t_final - plateau) / 2, t + plateau / 2, x)
        x = tf.where(np.abs(t - t_final / 2) < plateau / 2, (t_final) / 2, x)
        length = params["risefall"].get_value() * 2
    else:
        x = tf.identity(t)
        length = tf.identity(width)
    shape = tf.zeros_like(t)
    for n, coeff in enumerate(fourier_coeffs):
        shape += coeff * (
            1 - tf.cos(2 * np.pi * (n + 1) * (x - (t_final - length) / 2) / length)
        )
    if "sin_coeffs" in params:
        for n, coeff in enumerate(params["sin_coeffs"].get_value()):
            shape += coeff * (
                tf.sin((np.pi * (2 * n + 1)) * (x - (t_final - length) / 2) / length)
            )
    shape = tf.where(tf.abs(t_final / 2 - t) > width / 2, tf.zeros_like(t), shape)
    shape /= tf.reduce_max(shape)
    shape = shape * (1 - offset / amp) + offset / amp
    return shape


@env_reg_deco
def flattop_risefall_1ns(t, params):
    """Flattop gaussian with fixed width of 1ns."""
    params["risefall"] = 1e-9
    return flattop_risefall(t, params)


@env_reg_deco
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
    t_final = tf.cast(params["t_final"].get_value(), tf.float64)
    sigma = tf.cast(params["sigma"].get_value(), tf.float64)
    gauss = tf.exp(-((t - t_final / 2) ** 2) / (2 * sigma ** 2))

    offset = tf.exp(-(t_final ** 2) / (8 * sigma ** 2))
    norm = (
        tf.sqrt(2 * np.pi * sigma ** 2) * tf.math.erf(t_final / (np.sqrt(8) * sigma))
        - t_final * offset
    )
    return (gauss - offset) / norm


@env_reg_deco
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
    params["sigma"] = Qty(
        value=params["t_final"].get_value() / 6,
        min_val=params["t_final"].get_value() / 8,
        max_val=params["t_final"].get_value() / 4,
        unit=params["t_final"].unit,
    )
    return gaussian_sigma(t, params)


@env_reg_deco
def cosine(t, params):
    """
    Cosine-shaped envelope. Maximum value is 1, area is given by length.

    Parameters
    ----------
    params : dict
        t_final : float
            Total length of the Gaussian.
        sigma: float
            Width of the Gaussian.

    """
    # TODO Add zeroes for t>t_final
    t_final = tf.cast(params["t_final"].get_value(), tf.float64)
    cos = 0.5 * (1 - tf.cos(2 * np.pi * t / t_final))
    return cos


@env_reg_deco
def cosine_flattop(t, params):
    """
    Cosine-shaped envelope. Maximum value is 1, area is given by length.

    Parameters
    ----------
    params : dict
        t_final : float
            Total length of the Gaussian.
        sigma: float
            Width of the Gaussian.

    """
    t_rise = tf.cast(params["t_rise"].get_value(), tf.float64)
    dt = t[1] - t[0]
    n_rise = tf.cast(t_rise / dt, tf.int32)
    n_flat = len(t) - 2 * n_rise
    cos_flt = tf.concat(
        [
            0.5 * (1 - tf.cos(np.pi * t[:n_rise] / t_rise)),
            tf.ones(n_flat, dtype=tf.float64),
            0.5 * (1 + tf.cos(np.pi * t[:n_rise] / t_rise)),
        ],
        axis=0,
    )
    return cos_flt


@env_reg_deco
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
    t_final = tf.cast(params["t_final"].get_value(), tf.float64)
    sigma = params["sigma"].get_value()
    gauss = tf.exp(-((t - t_final / 2) ** 2) / (2 * sigma ** 2))
    return gauss


@env_reg_deco
def gaussian_der_nonorm(t, params):
    """Derivative of the normalized gaussian (ifself not normalized)."""
    t_final = tf.cast(params["t_final"].get_value(), tf.float64)
    sigma = tf.cast(params["sigma"].get_value(), tf.float64)
    gauss_der = (
        tf.exp(-((t - t_final / 2) ** 2) / (2 * sigma ** 2))
        * (t - t_final / 2)
        / sigma ** 2
    )
    return gauss_der


@env_reg_deco
def gaussian_der(t, params):
    """Derivative of the normalized gaussian (ifself not normalized)."""
    t_final = tf.cast(params["t_final"].get_value(), tf.float64)
    sigma = tf.cast(params["sigma"].get_value(), tf.float64)
    gauss_der = (
        tf.exp(-((t - t_final / 2) ** 2) / (2 * sigma ** 2))
        * (t - t_final / 2)
        / sigma ** 2
    )
    norm = tf.sqrt(2 * np.pi * sigma ** 2) * tf.math.erf(
        t_final / (tf.sqrt(8) * sigma)
    ) - t_final * tf.exp(-(t_final ** 2) / (8 * sigma ** 2))
    return gauss_der / norm


@env_reg_deco
def drag_sigma(t, params):
    """Second order gaussian."""
    t_final = tf.cast(params["t_final"].get_value(), tf.float64)
    sigma = tf.cast(params["sigma"].get_value(), tf.float64)
    drag = tf.exp(-((t - t_final / 2) ** 2) / (2 * sigma ** 2))
    norm = tf.sqrt(2 * np.pi * sigma ** 2) * tf.math.erf(
        t_final / (np.sqrt(8) * sigma)
    ) - t_final * tf.exp(-(t_final ** 2) / (8 * sigma ** 2))
    offset = tf.exp(-(t_final ** 2) / (8 * sigma ** 2))
    return (drag - offset) ** 2 / norm


@env_reg_deco
def drag(t, params):
    """Second order gaussian with fixed time/sigma ratio."""
    DeprecationWarning("Using standard width. Better use drag_sigma.")
    params["sigma"] = Qty(
        value=params["t_final"].get_value() / 4,
        min_val=params["t_final"].get_value() / 8,
        max_val=params["t_final"].get_value() / 2,
        unit=params["t_final"].unit,
    )
    return drag_sigma(t, params)


@env_reg_deco
def drag_der(t, params):
    """Derivative of second order gaussian."""
    t_final = tf.cast(params["t_final"].get_value(), tf.float64)
    sigma = tf.cast(params["sigma"].get_value(), tf.float64)
    norm = tf.sqrt(2 * np.pi * sigma ** 2) * tf.math.erf(
        t_final / (np.sqrt(8) * sigma)
    ) - t_final * tf.exp(-(t_final ** 2) / (8 * sigma ** 2))
    offset = tf.exp(-(t_final ** 2) / (8 * sigma ** 2))
    der = (
        -2
        * (tf.exp(-((t - t_final / 2) ** 2) / (2 * sigma ** 2)) - offset)
        * (np.exp(-((t - t_final / 2) ** 2) / (2 * sigma ** 2)))
        * (t - t_final / 2)
        / sigma ** 2
        / norm
    )
    return der


@env_reg_deco
def flattop_variant(t, params):
    """
    Flattop variant.
    """
    t_up = params["t_up"]
    t_down = params["t_down"]
    ramp = params["ramp"]
    value = np.ones(len(t))
    if ramp > (t_down - t_up) / 2:
        ramp = (t_down - t_up) / 2
    sigma = np.sqrt(2) * ramp * 0.2
    for i, e in enumerate(t):
        if t_up <= e <= t_up + ramp:
            value[i] = np.exp(-((e - t_up - ramp) ** 2) / (2 * sigma ** 2))
        elif t_up + ramp < e < t_down - ramp:
            value[i] = 1
        elif t_down >= e >= t_down - ramp:
            value[i] = np.exp(-((e - t_down + ramp) ** 2) / (2 * sigma ** 2))
        else:
            value[i] = 0
    return value
