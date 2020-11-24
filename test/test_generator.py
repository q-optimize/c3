""" Testing module for the generator and devices classes"""

import pickle
import numpy as np
from c3.generator.devices import LO, AWG, Mixer, Response, Digital_to_Analog, Volts_to_Hertz
from c3.generator.generator import Generator
from c3.signal.gates import Instruction
from c3.signal.pulse import Envelope, Carrier
from c3.c3objs import Quantity
import c3.libraries.envelopes as env_lib

sim_res = 100e9  # Resolution for numerical simulation
awg_res = 2e9  # Realistic, limited resolution of an AWG
lo = LO(name='lo', resolution=sim_res)
awg = AWG(name='awg', resolution=awg_res)
mixer = Mixer(name='mixer')

resp = Response(
    name='resp',
    rise_time=Quantity(
        value=0.3e-9,
        min=0.05e-9,
        max=0.6e-9,
        unit='s'
    ),
    resolution=sim_res
)

dig_to_an = Digital_to_Analog(
    name="dac",
    resolution=sim_res
)

v2hz = 1e9
v_to_hz = Volts_to_Hertz(
    name='v_to_hz',
    V_to_Hz=Quantity(
        value=v2hz,
        min=0.9e9,
        max=1.1e9,
        unit='Hz 2pi/V'
    )
)

generator = Generator([lo, awg, mixer, v_to_hz, dig_to_an, resp])

t_final = 7e-9   # Time for single qubit gates
sideband = 50e6 * 2 * np.pi
gauss_params_single = {
    'amp': Quantity(
        value=0.5,
        min=0.4,
        max=0.6,
        unit="V"
    ),
    't_final': Quantity(
        value=t_final,
        min=0.5 * t_final,
        max=1.5 * t_final,
        unit="s"
    ),
    'sigma': Quantity(
        value=t_final / 4,
        min=t_final / 8,
        max=t_final / 2,
        unit="s"
    ),
    'xy_angle': Quantity(
        value=0.0,
        min=-0.5 * np.pi,
        max=2.5 * np.pi,
        unit='rad'
    ),
    'freq_offset': Quantity(
        value=-sideband - 3e6 * 2 * np.pi,
        min=-56 * 1e6 * 2 * np.pi,
        max=-52 * 1e6 * 2 * np.pi,
        unit='Hz 2pi'
    ),
    'delta': Quantity(
        value=-1,
        min=-5,
        max=3,
        unit=""
    )
}
gauss_env_single = Envelope(
    name="gauss",
    desc="Gaussian comp for single-qubit gates",
    params=gauss_params_single,
    shape=env_lib.gaussian_nonorm
)


lo_freq_q1 = 5e9 * 2 * np.pi + sideband
carrier_parameters = {
    'freq': Quantity(
        value=lo_freq_q1,
        min=4.5e9 * 2 * np.pi,
        max=6e9 * 2 * np.pi,
        unit='Hz 2pi'
    ),
    'framechange': Quantity(
        value=0.0,
        min= -np.pi,
        max= 3 * np.pi,
        unit='rad'
    )
}
carr = Carrier(
    name="carrier",
    desc="Frequency of the local oscillator",
    params=carrier_parameters
)

X90p_q1 = Instruction(
    name="X90p",
    t_start=0.0,
    t_end=t_final,
    channels=["d1"]
)
X90p_q1.add_component(gauss_env_single, "d1")
X90p_q1.add_component(carr, "d1")

tstart = X90p_q1.t_start
tend = X90p_q1.t_end
chan = "d1"

with open("test/generator_data.pickle", "rb") as filename:
    data = pickle.load(filename)


def test_LO() -> None:
    lo_sig = lo.create_signal(X90p_q1.comps["d1"], tstart, tend)[0]
    assert (lo_sig["values"][0].numpy() == data["lo_sig"]["values"][0].numpy()).all()
    assert (lo_sig["values"][1].numpy() == data["lo_sig"]["values"][1].numpy()).all()
    assert (lo_sig["ts"].numpy() == data["lo_sig"]["ts"].numpy()).all()


def test_AWG() -> None:
    awg_sig = awg.create_IQ("d1", X90p_q1.comps["d1"], tstart, tend)
    assert (awg_sig["inphase"].numpy() == data["awg_sig"]["inphase"].numpy()).all()
    assert (awg_sig["quadrature"].numpy() == data["awg_sig"]["quadrature"].numpy()).all()


def test_DAC() -> None:
    dac_sig = dig_to_an.resample(data["awg_sig"], tstart, tend)
    assert (dac_sig["inphase"].numpy() == data["dig_to_an_sig"]["inphase"].numpy()).all()
    assert (dac_sig["quadrature"].numpy() == data["dig_to_an_sig"]["quadrature"].numpy()).all()


def test_Response() -> None:
    resp_sig = resp.process(data["dig_to_an_sig"])
    assert (resp_sig["inphase"].numpy() == data["resp_sig"]["inphase"].numpy()).all()
    assert (resp_sig["quadrature"].numpy() == data["resp_sig"]["quadrature"].numpy()).all()


def test_mixer() -> None:
    mixed_sig = mixer.combine(data["lo_sig"], data["resp_sig"])
    assert (mixed_sig.numpy() == data["mixer_sig"].numpy()).all()


def test_v2hz() -> None:
    final_sig = v_to_hz.transform(data["mixer_sig"], 0)
    assert (final_sig.numpy() == data["v2hz_sig"].numpy()).all()


def test_full_signal_chain() -> None:
    full_signal = generator.generate_signals(X90p_q1)
    assert (full_signal[0]["d1"]["values"].numpy() == data["full_signal"][0]["d1"]["values"].numpy()).all()
    assert (full_signal[1].numpy() == data["full_signal"][1].numpy()).all()
