import pickle

from c3.c3objs import Quantity
from c3.libraries.envelopes import envelopes
import numpy as np
import pytest

ts = np.linspace(0, 10e-9, 100)

with open("test/envelopes.pickle", "rb") as filename:
    test_data = pickle.load(filename)

ABS_TOL_FACTOR = 1e-11


def get_atol(test_type: str) -> float:
    """Get the absolute tolerance corresponding to a specific test data

    Parameters
    ----------
    test_type : str
        String representing the test type to be used as a key in the test_data dict

    Returns
    -------
    float
        Absolute tolerance for the desired value of this test type
    """
    return ABS_TOL_FACTOR * np.max(test_data[test_type])


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
        atol=get_atol("pwc_shape"),
    )

    np.testing.assert_allclose(
        actual=envelopes["pwc_symmetric"](t=ts, params=params),
        desired=test_data["pwc_symmetric"],
        atol=get_atol("pwc_symmetric"),
    )

    np.testing.assert_allclose(
        actual=envelopes["pwc_shape_plateau"](t=ts, params=params),
        desired=test_data["pwc_shape_plateau1"],
        atol=get_atol("pwc_shape_plateau1"),
    )

    params["width"] = Quantity(5e-9)
    np.testing.assert_allclose(
        actual=envelopes["pwc_shape_plateau"](t=ts, params=params),
        desired=test_data["pwc_shape_plateau2"],
        atol=get_atol("pwc_shape_plateau2"),
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
        atol=get_atol("delta_pulse"),
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
        atol=get_atol("fourier_sin"),
    )

    np.testing.assert_allclose(
        actual=envelopes["fourier_cos"](t=ts, params=params),
        desired=test_data["fourier_cos"],
        atol=get_atol("fourier_cos"),
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
        atol=get_atol("slepian_fourier"),
    )

    params["risefall"] = Quantity(4e-9)
    np.testing.assert_allclose(
        actual=envelopes["slepian_fourier"](t=ts, params=params),
        desired=test_data["slepian_fourier_risefall"],
        atol=get_atol("slepian_fourier_risefall"),
    )

    params["sin_coeffs"] = Quantity([0.3])
    np.testing.assert_allclose(
        actual=envelopes["slepian_fourier"](t=ts, params=params),
        desired=test_data["slepian_fourier_sin"],
        atol=get_atol("slepian_fourier_sin"),
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
        atol=get_atol("trapezoid"),
    )

    np.testing.assert_allclose(
        actual=envelopes["flattop_risefall"](t=ts, params=params),
        desired=test_data["flattop_risefall"],
        atol=get_atol("flattop_risefall"),
    )

    np.testing.assert_allclose(
        actual=envelopes["flattop_risefall_1ns"](t=ts, params=params),
        desired=test_data["flattop_risefall_1ns"],
        atol=get_atol("flattop_risefall_1ns"),
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
        atol=get_atol("flattop_variant"),
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
        atol=get_atol("flattop_cut"),
    )

    params = {
        "risefall": Quantity(2e-9),
        "width": Quantity(9e-9),
        "t_final": Quantity(10e-9),
    }

    np.testing.assert_allclose(
        actual=envelopes["flattop_cut_center"](t=ts, params=params),
        desired=test_data["flattop_cut_center"],
        atol=get_atol("flattop_cut_center"),
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
        atol=get_atol("gaussian_sigma"),
    )

    np.testing.assert_allclose(
        actual=envelopes["gaussian"](t=ts, params=params),
        desired=test_data["gaussian"],
        atol=get_atol("gaussian"),
    )

    np.testing.assert_allclose(
        actual=envelopes["gaussian_nonorm"](t=ts, params=params),
        desired=test_data["gaussian_nonorm"],
        atol=get_atol("gaussian_nonorm"),
    )

    np.testing.assert_allclose(
        actual=envelopes["gaussian_der_nonorm"](t=ts, params=params),
        desired=test_data["gaussian_der_nonorm"],
        atol=get_atol("gaussian_der_nonorm"),
    )

    np.testing.assert_allclose(
        actual=envelopes["gaussian_der"](t=ts, params=params),
        desired=test_data["gaussian_der"],
        atol=get_atol("gaussian_der"),
    )

    np.testing.assert_allclose(
        actual=envelopes["drag_sigma"](t=ts, params=params),
        desired=test_data["drag_sigma"],
        atol=get_atol("drag_sigma"),
    )

    np.testing.assert_allclose(
        actual=envelopes["drag_der"](t=ts, params=params),
        desired=test_data["drag_der"],
        atol=get_atol("drag_der"),
    )

    np.testing.assert_allclose(
        actual=envelopes["drag"](t=ts, params=params),
        desired=test_data["drag"],
        atol=get_atol("drag"),
    )


@pytest.mark.unit
def test_cosine():
    params = {
        "t_final": Quantity(10e-9),
    }

    np.testing.assert_allclose(
        actual=envelopes["cosine"](t=ts, params=params),
        desired=test_data["cosine"],
        atol=get_atol("cosine"),
    )

    params = {
        "t_final": Quantity(10e-9),
        "t_rise": Quantity(2e-9),
    }

    np.testing.assert_allclose(
        actual=np.reshape(
            envelopes["cosine_flattop"](t=np.reshape(ts, (-1, 1)), params=params), (-1,)
        ),
        desired=test_data["cosine_flattop"],
        atol=get_atol("cosine_flattop"),
    )
    # Nico: to be consistent with the signal generation code, the resphapes above are necessary. Somewhere in the
    # masking the time vector gets an additional dimension. It all works fine since it's elementwise, but the
    # flattop implementation has a concat in it that is strict about shapes. This should be investigated.


@pytest.mark.unit
def test_nodrive():
    params = {}
    np.testing.assert_allclose(
        actual=envelopes["no_drive"](t=ts, params=params),
        desired=test_data["no_drive"],
        atol=get_atol("no_drive"),
    )

    np.testing.assert_allclose(
        actual=envelopes["rect"](t=ts, params=params),
        desired=test_data["rect"],
        atol=get_atol("rect"),
    )
