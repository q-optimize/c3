"""
testing module ParameterMap class
"""

import copy
import numpy as np
import pytest

import c3.libraries.envelopes as envelopes

from c3.parametermap import ParameterMap, Quantity
from c3.signal.pulse import Envelope, Carrier
from c3.signal.gates import Instruction


def setup_pmap() -> ParameterMap:
    t_final = 7e-9  # Time for single qubit gates
    sideband = 50e6
    lo_freq = 5e9 + sideband

    # ### MAKE GATESET
    gauss_params_single = {
        "amp": Quantity(value=0.45, min_val=0.4, max_val=0.6, unit="V"),
        "t_final": Quantity(
            value=t_final, min_val=0.5 * t_final, max_val=1.5 * t_final, unit="s"
        ),
        "sigma": Quantity(
            value=t_final / 4, min_val=t_final / 8, max_val=t_final / 2, unit="s"
        ),
        "xy_angle": Quantity(
            value=0.0, min_val=-0.5 * np.pi, max_val=2.5 * np.pi, unit="rad"
        ),
        "freq_offset": Quantity(
            value=-sideband - 0.5e6, min_val=-53 * 1e6, max_val=-47 * 1e6, unit="Hz 2pi"
        ),
        "delta": Quantity(value=-1, min_val=-5, max_val=3, unit=""),
    }

    gauss_env_single = Envelope(
        name="gauss",
        desc="Gaussian comp for single-qubit gates",
        params=gauss_params_single,
        shape=envelopes.gaussian_nonorm,
    )
    nodrive_env = Envelope(
        name="no_drive",
        params={
            "t_final": Quantity(
                value=t_final, min_val=0.5 * t_final, max_val=1.5 * t_final, unit="s"
            )
        },
        shape=envelopes.no_drive,
    )
    carrier_parameters = {
        "freq": Quantity(value=lo_freq, min_val=4.5e9, max_val=6e9, unit="Hz 2pi"),
        "framechange": Quantity(
            value=0.0, min_val=-np.pi, max_val=3 * np.pi, unit="rad"
        ),
    }
    carr = Carrier(
        name="carrier",
        desc="Frequency of the local oscillator",
        params=carrier_parameters,
    )

    X90p = Instruction(name="X90p", t_start=0.0, t_end=t_final, channels=["d1"])
    QId = Instruction(name="Id", t_start=0.0, t_end=t_final, channels=["d1"])

    X90p.add_component(gauss_env_single, "d1")
    X90p.add_component(carr, "d1")
    QId.add_component(nodrive_env, "d1")
    QId.add_component(copy.deepcopy(carr), "d1")
    QId.comps["d1"]["carrier"].params["framechange"].set_value(
        (-sideband * t_final) % (2 * np.pi)
    )
    Y90p = copy.deepcopy(X90p)
    Y90p.name = "Y90p"
    X90m = copy.deepcopy(X90p)
    X90m.name = "X90m"
    Y90m = copy.deepcopy(X90p)
    Y90m.name = "Y90m"
    Y90p.comps["d1"]["gauss"].params["xy_angle"].set_value(0.5 * np.pi)
    X90m.comps["d1"]["gauss"].params["xy_angle"].set_value(np.pi)
    Y90m.comps["d1"]["gauss"].params["xy_angle"].set_value(1.5 * np.pi)

    parameter_map = ParameterMap(instructions=[QId, X90p, Y90p, X90m, Y90m])

    gateset_opt_map = [
        [
            ("X90p", "d1", "gauss", "amp"),
            ("Y90p", "d1", "gauss", "amp"),
            ("X90m", "d1", "gauss", "amp"),
            ("Y90m", "d1", "gauss", "amp"),
        ],
        [
            ("X90p", "d1", "gauss", "delta"),
            ("Y90p", "d1", "gauss", "delta"),
            ("X90m", "d1", "gauss", "delta"),
            ("Y90m", "d1", "gauss", "delta"),
        ],
        [
            ("X90p", "d1", "gauss", "freq_offset"),
            ("Y90p", "d1", "gauss", "freq_offset"),
            ("X90m", "d1", "gauss", "freq_offset"),
            ("Y90m", "d1", "gauss", "freq_offset"),
        ],
        [("Id", "d1", "carrier", "framechange")],
    ]

    parameter_map.set_opt_map(gateset_opt_map)

    return parameter_map


pmap = setup_pmap()


@pytest.mark.unit
def test_parameter_print() -> None:
    """
    Check parameter printing.
    """
    pmap.print_parameters()


@pytest.mark.unit
def test_parameter_str() -> None:
    """
    Check casting to string.
    """
    str(pmap)


@pytest.mark.unit
def test_parameter_get() -> None:
    """
    Check that four parameters are set.
    """
    assert len(pmap.get_parameters()) == 4


@pytest.mark.unit
def test_parameter_get_value() -> None:
    """
    Check that four parameters are set.
    """
    assert str(pmap.get_parameter(("X90p", "d1", "gauss", "amp"))) == "450.000 mV "


@pytest.mark.unit
def test_parameter_equiv() -> None:
    """
    Check that two equivalent parameters do not point to the same memory address.
    """
    amp1 = pmap.get_parameter(("X90p", "d1", "gauss", "amp"))
    amp2 = pmap.get_parameter(("Y90p", "d1", "gauss", "amp"))
    assert amp1 is not amp2


@pytest.mark.unit
def test_parameter_set_equiv() -> None:
    """
    Check that setting equivalent parameters also sets the other one.
    """
    amp_ids = [[("X90p", "d1", "gauss", "amp"), ("X90m", "d1", "gauss", "amp")]]
    pmap.set_parameters([0.55], amp_ids)
    amp1 = pmap.get_parameter(("X90p", "d1", "gauss", "amp"))
    amp2 = pmap.get_parameter(("X90m", "d1", "gauss", "amp"))
    assert amp1.get_value() == amp2.get_value()


@pytest.mark.unit
def test_parameter_set_indepentent() -> None:
    """
    Check that setting equivalent parameters also sets the other one.
    """
    amp_ids = [[("X90p", "d1", "gauss", "amp")], [("X90m", "d1", "gauss", "amp")]]
    pmap.set_parameters([0.55, 0.41], amp_ids)
    amp1 = pmap.get_parameter(("X90p", "d1", "gauss", "amp"))
    amp2 = pmap.get_parameter(("X90m", "d1", "gauss", "amp"))
    assert amp1.get_value() != amp2.get_value()


@pytest.mark.unit
def test_parameter_set_opt() -> None:
    """
    Test the setting in optimizer format.
    """
    amp_ids = [[("X90p", "d1", "gauss", "amp")], [("X90m", "d1", "gauss", "amp")]]
    pmap.set_opt_map(amp_ids)
    pmap.set_parameters_scaled([-1.0, 1.0])  # -+1 correspond to min and max allowd
    assert pmap.get_parameter(("X90p", "d1", "gauss", "amp")).get_value() == 0.4
    assert pmap.get_parameter(("X90m", "d1", "gauss", "amp")).get_value() == 0.6
