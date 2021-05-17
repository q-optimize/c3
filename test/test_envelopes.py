import pickle

from c3.c3objs import Quantity
from c3.libraries.envelopes import envelopes
import numpy as np
import pytest

ts = np.linspace(0, 10e-9, 100)

with open("test/envelopes.pickle", "rb") as filename:
    test_data = pickle.load(filename)


@pytest.mark.unit
def test_pwc_shape():
    params = {
        "t_bin_start": Quantity(1e-10),
        "t_bin_end": Quantity(9.9e-9),
        "t_final": Quantity(10e-9),
        "inphase": Quantity([0, 0.1, 0.3, 0.5, 0.1, 1.1, 0.4, 0.1]),
    }
    np.testing.assert_allclose(
        actual=envelopes["pwc_shape"](t=ts, params=params),
        desired=test_data["pwc_shape"],
    )

    np.testing.assert_allclose(
        actual=envelopes["pwc_symmetric"](t=ts, params=params),
        desired=test_data["pwc_symmetric"],
    )

    np.testing.assert_allclose(
        actual=envelopes["pwc_shape_plateau"](t=ts, params=params),
        desired=test_data["pwc_shape_plateau1"],
    )

    params["width"] = Quantity(5e-9)
    np.testing.assert_allclose(
        actual=envelopes["pwc_shape_plateau"](t=ts, params=params),
        desired=test_data["pwc_shape_plateau2"],
    )


@pytest.mark.unit
def test_delta_pulse():
    params = {
        "t_sig": Quantity(
            [
                0.5e-9,
            ]
        ),
        "t_final": Quantity(10e-9),
    }
    np.testing.assert_allclose(
        actual=envelopes["delta_pulse"](t=ts, params=params),
        desired=test_data["delta_pulse"],
    )


@pytest.mark.unit
def test_fourier():
    params = {
        "amps": Quantity([0.5, 0.2]),
        "freqs": Quantity([1e6, 1e10]),
        "phases": Quantity([0, 1]),
        "t_final": Quantity(10e-9),
    }

    np.testing.assert_allclose(
        actual=envelopes["fourier_sin"](t=ts, params=params),
        desired=test_data["fourier_sin"],
    )

    np.testing.assert_allclose(
        actual=envelopes["fourier_cos"](t=ts, params=params),
        desired=test_data["fourier_cos"],
    )

    params = {
        "width": Quantity(9e-9),
        "fourier_coeffs": Quantity([1, 0.5, 0.2]),
        "offset": Quantity(0.1),
        "amp": Quantity(0.5),
        "t_final": Quantity(10e-9),
    }

    np.testing.assert_allclose(
        actual=envelopes["slepian_fourier"](t=ts, params=params),
        desired=test_data["slepian_fourier"],
    )

    params["risefall"] = Quantity(4e-9)
    np.testing.assert_allclose(
        actual=envelopes["slepian_fourier"](t=ts, params=params),
        desired=test_data["slepian_fourier_risefall"],
    )

    params["sin_coeffs"] = Quantity([0.3])
    np.testing.assert_allclose(
        actual=envelopes["slepian_fourier"](t=ts, params=params),
        desired=test_data["slepian_fourier_sin"],
    )


@pytest.mark.unit
def test_flattop():
    params = {
        "risefall": Quantity(2e-9),
        "t_final": Quantity(10e-9),
    }

    np.testing.assert_allclose(
        actual=envelopes["trapezoid"](t=ts, params=params),
        desired=test_data["trapezoid"],
    )

    np.testing.assert_allclose(
        actual=envelopes["flattop_risefall"](t=ts, params=params),
        desired=test_data["flattop_risefall"],
    )

    np.testing.assert_allclose(
        actual=envelopes["flattop_risefall_1ns"](t=ts, params=params),
        desired=test_data["flattop_risefall_1ns"],
    )

    params = {
        "ramp": Quantity(2e-9),
        "t_up": Quantity(1e-9),
        "t_down": Quantity(10e-9),
        "t_final": Quantity(10e-9),
    }

    np.testing.assert_allclose(
        actual=envelopes["flattop_variant"](t=ts, params=params),
        desired=test_data["flattop_variant"],
    )

    params = {
        "risefall": Quantity(2e-9),
        "t_up": Quantity(1e-9),
        "t_down": Quantity(10e-9),
        "t_final": Quantity(10e-9),
    }

    np.testing.assert_allclose(
        actual=envelopes["flattop"](t=ts, params=params), desired=test_data["flattop"]
    )


@pytest.mark.unit
def test_flattop_cut():
    params = {
        "risefall": Quantity(2e-9),
        "t_up": Quantity(1e-9),
        "t_down": Quantity(10e-9),
        "t_final": Quantity(10e-9),
    }

    np.testing.assert_allclose(
        actual=envelopes["flattop_cut"](t=ts, params=params),
        desired=test_data["flattop_cut"],
    )

    params = {
        "risefall": Quantity(2e-9),
        "width": Quantity(9e-9),
        "t_final": Quantity(10e-9),
    }

    np.testing.assert_allclose(
        actual=envelopes["flattop_cut_center"](t=ts, params=params),
        desired=test_data["flattop_cut_center"],
    )


@pytest.mark.unit
def test_gaussian():
    params = {
        "t_final": Quantity(10e-9),
        "sigma": Quantity(5e-9),
    }

    np.testing.assert_allclose(
        actual=envelopes["gaussian_sigma"](t=ts, params=params),
        desired=test_data["gaussian_sigma"],
    )

    np.testing.assert_allclose(
        actual=envelopes["gaussian"](t=ts, params=params), desired=test_data["gaussian"]
    )

    np.testing.assert_allclose(
        actual=envelopes["gaussian_nonorm"](t=ts, params=params),
        desired=test_data["gaussian_nonorm"],
    )

    np.testing.assert_allclose(
        actual=envelopes["gaussian_der_nonorm"](t=ts, params=params),
        desired=test_data["gaussian_der_nonorm"],
    )

    np.testing.assert_allclose(
        actual=envelopes["gaussian_der"](t=ts, params=params),
        desired=test_data["gaussian_der"],
    )

    np.testing.assert_allclose(
        actual=envelopes["drag_sigma"](t=ts, params=params),
        desired=test_data["drag_sigma"],
    )

    np.testing.assert_allclose(
        actual=envelopes["drag_der"](t=ts, params=params), desired=test_data["drag_der"]
    )

    np.testing.assert_allclose(
        actual=envelopes["drag"](t=ts, params=params), desired=test_data["drag"]
    )


@pytest.mark.unit
def test_cosine():
    params = {
        "t_final": Quantity(10e-9),
    }

    np.testing.assert_allclose(
        actual=envelopes["cosine"](t=ts, params=params), desired=test_data["cosine"]
    )

    params = {
        "t_final": Quantity(10e-9),
        "t_rise": Quantity(2e-9),
    }

    np.testing.assert_allclose(
        actual=envelopes["cosine_flattop"](t=ts, params=params),
        desired=test_data["cosine_flattop"],
    )


@pytest.mark.unit
def test_nodrive():
    params = {}
    np.testing.assert_allclose(
        actual=envelopes["no_drive"](t=ts, params=params), desired=test_data["no_drive"]
    )

    np.testing.assert_allclose(
        actual=envelopes["rect"](t=ts, params=params), desired=test_data["rect"]
    )
