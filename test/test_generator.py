""" Testing module for the generator and devices classes"""

import pickle
import numpy as np
import pytest
from c3.generator.devices import (
    LO,
    AWG,
    Mixer,
    ResponseFFT,
    DigitalToAnalog,
    VoltsToHertz,
    Crosstalk,
)
from c3.generator.generator import Generator
from c3.signal.gates import Instruction
from c3.signal.pulse import Envelope, Carrier
from c3.c3objs import Quantity
import c3.libraries.envelopes as env_lib

sim_res = 100e9  # Resolution for numerical simulation
awg_res = 2e9  # Realistic, limited resolution of an AWG

lo = LO(name="lo", resolution=sim_res, outputs=1)
awg = AWG(name="awg", resolution=awg_res, outputs=1)
dac = DigitalToAnalog(name="dac", resolution=sim_res, inputs=1, outputs=1)
resp = ResponseFFT(
    name="resp",
    rise_time=Quantity(value=0.3e-9, min_val=0.05e-9, max_val=0.6e-9, unit="s"),
    resolution=sim_res,
    inputs=1,
    outputs=1,
)
mixer = Mixer(name="mixer", inputs=2, outputs=1)
v_to_hz = VoltsToHertz(
    name="v_to_hz",
    V_to_Hz=Quantity(value=1e9, min_val=0.9e9, max_val=1.1e9, unit="Hz/V"),
    inputs=1,
    outputs=1,
)
xtalk = Crosstalk(
    name="crosstalk",
    channels=["d1", "d2"],
    crosstalk_matrix=Quantity(
        value=[[1, 0], [1, 0]], min_val=[[0, 0], [0, 0]], max_val=[[1, 1], [1, 1]]
    ),
)

generator = Generator(
    devices={
        "LO": lo,
        "AWG": awg,
        "DigitalToAnalog": dac,
        "Response": resp,
        "Mixer": mixer,
        "VoltsToHertz": v_to_hz,
    },
    chains={
        "d1": {
            "LO": [],
            "AWG": [],
            "DigitalToAnalog": ["AWG"],
            "Response": ["DigitalToAnalog"],
            "Mixer": ["LO", "Response"],
            "VoltsToHertz": ["Mixer"],
        },
    },
)

t_final = 7e-9  # Time for single qubit gates
sideband = 50e6
gauss_params_single = {
    "amp": Quantity(value=0.5, min_val=0.4, max_val=0.6, unit="V"),
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
        value=-sideband - 3e6, min_val=-56 * 1e6, max_val=-52 * 1e6, unit="Hz 2pi"
    ),
    "delta": Quantity(value=-1, min_val=-5, max_val=3, unit=""),
}
gauss_env_single = Envelope(
    name="gauss",
    desc="Gaussian comp for single-qubit gates",
    params=gauss_params_single,
    shape=env_lib.gaussian_nonorm,
)


lo_freq_q1 = 5e9 + sideband
carrier_parameters = {
    "freq": Quantity(value=lo_freq_q1, min_val=4.5e9, max_val=6e9, unit="Hz 2pi"),
    "framechange": Quantity(value=0.0, min_val=-np.pi, max_val=3 * np.pi, unit="rad"),
}
carr = Carrier(
    name="carrier", desc="Frequency of the local oscillator", params=carrier_parameters
)

rx90p_q1 = Instruction(name="rx90p", t_start=0.0, t_end=t_final, channels=["d1"])
rx90p_q1.add_component(gauss_env_single, "d1")
rx90p_q1.add_component(carr, "d1")

tstart = rx90p_q1.t_start
tend = rx90p_q1.t_end
chan = "d1"

with open("test/generator_data.pickle", "rb") as filename:
    data = pickle.load(filename)


@pytest.mark.unit
def test_LO() -> None:
    lo_sig = lo.process(rx90p_q1, "d1", [])
    assert (
        lo_sig["inphase"].numpy() - data["lo_sig"]["values"][0].numpy() < 1e-12
    ).all()
    assert (
        lo_sig["quadrature"].numpy() - data["lo_sig"]["values"][1].numpy() < 1e-12
    ).all()
    assert (lo_sig["ts"].numpy() == data["lo_sig"]["ts"].numpy()).all()


@pytest.mark.unit
def test_AWG() -> None:
    awg_sig = awg.process(rx90p_q1, "d1", [])
    assert (
        awg_sig["inphase"].numpy() - data["awg_sig"]["inphase"].numpy() < 1e-12
    ).all()
    assert (
        awg_sig["quadrature"].numpy() - data["awg_sig"]["quadrature"].numpy() < 1e-12
    ).all()


@pytest.mark.unit
def test_DAC() -> None:
    dac_sig = dac.process(rx90p_q1, "d1", [data["awg_sig"]])
    assert (
        dac_sig["inphase"].numpy() - data["dig_to_an_sig"]["inphase"].numpy() < 1e-12
    ).all()
    assert (
        dac_sig["quadrature"].numpy() - data["dig_to_an_sig"]["quadrature"].numpy()
        < 1e-12
    ).all()


@pytest.mark.unit
def test_Response() -> None:
    resp_sig = resp.process(rx90p_q1, "d1", [data["dig_to_an_sig"]])
    assert (
        resp_sig["inphase"].numpy() - data["resp_sig"]["inphase"].numpy() < 1e-12
    ).all()
    assert (
        resp_sig["quadrature"].numpy() - data["resp_sig"]["quadrature"].numpy() < 1e-12
    ).all()


@pytest.mark.unit
def test_mixer() -> None:
    # For compatiblity with old dataset we need to wrap the LO output
    lo_signal = {
        "inphase": data["lo_sig"]["values"][0],
        "quadrature": data["lo_sig"]["values"][1],
        "ts": data["lo_sig"]["ts"],
    }
    mixed_sig = mixer.process(rx90p_q1, "d1", [lo_signal, data["resp_sig"]])
    assert (mixed_sig["values"].numpy() - data["mixer_sig"].numpy() < 1e-12).all()


@pytest.mark.unit
def test_v2hz() -> None:
    mixer_sig = {"values": data["mixer_sig"], "ts": data["lo_sig"]["ts"]}
    final_sig = v_to_hz.process(rx90p_q1, "d1", [mixer_sig])
    assert (final_sig["values"].numpy() - data["v2hz_sig"].numpy() < 1).all()


@pytest.mark.integration
def test_full_signal_chain() -> None:
    full_signal = generator.generate_signals(rx90p_q1)
    assert (
        full_signal["d1"]["values"].numpy()
        - data["full_signal"][0]["d1"]["values"].numpy()
        < 1
    ).all()


@pytest.mark.integration
def test_crosstalk() -> None:
    generator = Generator(
        devices={
            "LO": lo,
            "AWG": awg,
            "DigitalToAnalog": dac,
            "Response": resp,
            "Mixer": mixer,
            "VoltsToHertz": v_to_hz,
            "crosstalk": xtalk,
        },
        chains={
            "d1": {
                "LO": [],
                "AWG": [],
                "DigitalToAnalog": ["AWG"],
                "Response": ["DigitalToAnalog"],
                "Mixer": ["LO", "Response"],
                "VoltsToHertz": ["Mixer"],
            },
            "d2": {
                "LO": [],
                "AWG": [],
                "DigitalToAnalog": ["AWG"],
                "Response": ["DigitalToAnalog"],
                "Mixer": ["LO", "Response"],
                "VoltsToHertz": ["Mixer"],
            },
        },
    )
    RX90p_q1 = Instruction(
        name="RX90p", t_start=0.0, t_end=t_final, channels=["d1", "d2"]
    )
    RX90p_q1.add_component(gauss_env_single, "d1")
    RX90p_q1.add_component(carr, "d1")

    gauss_params_single_2 = {
        "amp": Quantity(value=0, min_val=-0.4, max_val=0.6, unit="V"),
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
            value=-sideband - 3e6, min_val=-56 * 1e6, max_val=-52 * 1e6, unit="Hz 2pi"
        ),
        "delta": Quantity(value=-1, min_val=-5, max_val=3, unit=""),
    }
    gauss_env_single_2 = Envelope(
        name="gauss",
        desc="Gaussian comp for single-qubit gates",
        params=gauss_params_single_2,
        shape=env_lib.gaussian_nonorm,
    )
    RX90p_q1.add_component(gauss_env_single_2, "d2")
    RX90p_q1.add_component(carr, "d2")
    full_signal = generator.generate_signals(RX90p_q1)
    assert (
        full_signal["d1"]["values"].numpy() == full_signal["d2"]["values"].numpy()
    ).all()
