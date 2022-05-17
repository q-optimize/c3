import numpy as np
import pytest
from c3.c3objs import Quantity
from c3.generator.generator import Generator
from c3.generator.devices import (
    LO,
    AWG,
    Mixer,
    DigitalToAnalog,
    VoltsToHertz,
)
from c3.signal.pulse import Envelope, Carrier
from c3.libraries.envelopes import pwc
from c3.signal.gates import Instruction

sim_res = 100e9  # Resolution for numerical simulation
awg_res = 2e9  # Realistic, limited resolution of an AWG

lo = LO(name="lo", resolution=sim_res, outputs=1)
awg = AWG(name="awg", resolution=awg_res, outputs=1)

dac = DigitalToAnalog(name="dac", resolution=sim_res, inputs=1, outputs=1)
mixer = Mixer(name="mixer", inputs=2, outputs=1)
v_to_hz = VoltsToHertz(
    name="v_to_hz",
    V_to_Hz=Quantity(value=1e9, min_val=0.9e9, max_val=1.1e9, unit="Hz/V"),
    inputs=1,
    outputs=1,
)

generator = Generator(
    devices={
        "LO": lo,
        "AWG": awg,
        "DigitalToAnalog": dac,
        "Mixer": mixer,
        "VoltsToHertz": v_to_hz,
    },
    chains={
        "d1": {
            "LO": [],
            "AWG": [],
            "DigitalToAnalog": ["AWG"],
            "Mixer": ["LO", "DigitalToAnalog"],
            "VoltsToHertz": ["Mixer"],
        },
    },
)

t_final = 7e-9  # Time for single qubit gates
slices = int(t_final * awg_res)

pwc_params = {
    "inphase": Quantity(value=np.random.randn(slices), unit="V"),
    "quadrature": Quantity(value=np.random.randn(slices), unit="V"),
    "amp": Quantity(value=1.0, unit="V"),
    "xy_angle": Quantity(
        value=0.0, min_val=-0.5 * np.pi, max_val=2.5 * np.pi, unit="rad"
    ),
    "freq_offset": Quantity(
        value=0, min_val=-5 * 1e6, max_val=5 * 1e6, unit="Hz 2pi"
    ),
}

pwc_env_single = Envelope(
    name="gauss",
    desc="Gaussian comp for single-qubit gates",
    params=pwc_params,
    shape=pwc
)

lo_freq_q1 = 5e9
carrier_parameters = {
    "freq": Quantity(value=lo_freq_q1, min_val=4.5e9, max_val=6e9, unit="Hz 2pi"),
    "framechange": Quantity(value=0.0, min_val=-np.pi, max_val=3 * np.pi, unit="rad"),
}
carr = Carrier(
    name="carrier", desc="Frequency of the local oscillator", params=carrier_parameters
)

pulse_gate = Instruction(name="pulse", t_start=0.0, t_end=t_final, channels=["d1"])
pulse_gate.add_component(pwc_env_single, "d1")
pulse_gate.add_component(carr, "d1")


@pytest.mark.unit
def test_generation() -> None:
    signal = generator.generate_signals(pulse_gate)
    assert signal["d1"]["values"] is not None
