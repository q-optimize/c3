"""Testing AWG modulating and phase shifting
"""
import pytest
import numpy as np
from c3.c3objs import Quantity as Qty
from c3.generator.generator import Generator
import c3.generator.devices as devices
from c3.signal.pulse import Envelope, Carrier
import c3.libraries.envelopes as envelopes
from c3.signal.gates import Instruction

AWG_RES = 100e9
SIM_RES = 100e9

generator = Generator(
    devices={
        "LO": devices.LO(name="lo", resolution=SIM_RES, outputs=1),
        "AWG": devices.AWG(name="awg", resolution=AWG_RES, outputs=1),
        "DigitalToAnalog": devices.DigitalToAnalog(
            name="dac", resolution=SIM_RES, inputs=1, outputs=1
        ),
        "Mixer": devices.Mixer(name="mixer", inputs=2, outputs=1),
    },
    chains={
        "d1": {
            "LO": [],
            "AWG": [],
            "DigitalToAnalog": ["AWG"],
            "Mixer": ["LO", "DigitalToAnalog"],
        },
    },
)

lo_freq_q1 = 2e9
t_final = 1 / lo_freq_q1


rect = Envelope(
    name="Rectangle",
    desc="",
    params={"t_final": Qty(t_final, "s")},
    shape=envelopes.rect,
)

carrier_parameters = {
    "freq": Qty(value=lo_freq_q1, min_val=1.5e9, max_val=6e9, unit="Hz 2pi"),
    "framechange": Qty(value=0.0, min_val=-np.pi, max_val=3 * np.pi, unit="rad"),
}
carr = Carrier(
    name="carrier", desc="Frequency of the local oscillator", params=carrier_parameters
)
rectangle = Instruction(name="Rectangle", t_start=0.0, t_end=t_final, channels=["d1"])
rectangle.add_component(rect, "d1")
rectangle.add_component(carr, "d1")


@pytest.mark.unit
def test_AWG_phase_shift() -> None:
    phase = 0.5
    rect.params["freq_offset"] = Qty(0, "Hz 2pi")
    rect.params["xy_angle"] = Qty(phase, "pi")

    sigs = generator.generate_signals(rectangle)
    correct_signal = np.cos(2 * np.pi * lo_freq_q1 * sigs["d1"]["ts"] - phase * np.pi)
    print(sigs["d1"]["values"])
    np.testing.assert_allclose(
        sigs["d1"]["values"].numpy(),
        correct_signal,
        atol=1e-9 * np.max(sigs["d1"]["values"]),
    )


@pytest.mark.unit
def test_AWG_freq_offset() -> None:
    offset = 53e6
    rect.params["freq_offset"] = Qty(offset, "Hz 2pi")
    rect.params["xy_angle"] = Qty(0, "pi")
    sigs = generator.generate_signals(rectangle)
    correct_signal = np.cos(2 * np.pi * (lo_freq_q1 + offset) * sigs["d1"]["ts"])
    assert (sigs["d1"]["values"].numpy() - correct_signal < 1e-9).all()
