import copy
import pickle

import hjson

from c3.experiment import Experiment
from c3.generator import devices
import numpy as np
from c3.c3objs import Quantity as Qty, hjson_decode, hjson_encode
from c3.generator.generator import Generator
from c3.libraries import hamiltonians
from c3.libraries.envelopes import envelopes
from c3.parametermap import ParameterMap
from c3.signal.gates import Instruction
from c3.signal.pulse import Envelope, Carrier
from c3.libraries.chip import TransmonExpanded, Transmon, Coupling
from c3.model import Model
import pytest

ATOL_SIG = 1e-8  # for comparing signals 10 nV
ATOL_FREQ = 1e3  # for comparing frequencies 1 KHz

freq_q1 = 5e9
freq_q2 = 6e9
fluxpoint1 = 0
fluxpoint2 = 0
phi_0 = 1
d1 = 0.1
d2 = 0.3
lvls1 = 6
lvls2 = 4
anhar1 = -200e6
anhar2 = -300e6
cut_excitations = 4

coupling_strength = 100e6
sim_res = 20e9
awg_res = 5e9

fluxamp = 0.3
cphase_time = 1e-9

q1 = TransmonExpanded(
    name="Qubit1",
    freq=Qty(value=freq_q1, min_val=0.0e9, max_val=10.0e9, unit="Hz 2pi"),
    phi=Qty(value=fluxpoint1, min_val=-5.0 * phi_0, max_val=5.0 * phi_0, unit="Phi0"),
    phi_0=Qty(value=phi_0, min_val=phi_0 * 0.9, max_val=phi_0 * 1.1, unit="Phi0"),
    d=Qty(value=d1, min_val=d1 * 0.9, max_val=d1 * 1.1, unit=""),
    hilbert_dim=lvls1,
    anhar=Qty(value=anhar1, min_val=-380e6, max_val=-120e6, unit="Hz 2pi"),
    # t1=Qty(value=t1_tc, min_val=1e-6, max_val=90e-6, unit="s"),
    # t2star=Qty(value=t2star_tc, min_val=1e-6, max_val=90e-6, unit="s"),
    # temp=Qty(value=init_temp, min_val=0.0, max_val=0.12, unit="K"),
)

q2 = Transmon(
    name="Qubit2",
    freq=Qty(value=freq_q2, min_val=0.0e9, max_val=10.0e9, unit="Hz 2pi"),
    phi=Qty(value=fluxpoint2, min_val=-5.0 * phi_0, max_val=5.0 * phi_0, unit="Phi0"),
    phi_0=Qty(value=phi_0, min_val=phi_0 * 0.9, max_val=phi_0 * 1.1, unit="Phi0"),
    d=Qty(value=d2, min_val=d2 * 0.9, max_val=d2 * 1.1, unit=""),
    hilbert_dim=lvls2,
    anhar=Qty(value=anhar2, min_val=-380e6, max_val=-120e6, unit="Hz 2pi"),
    # t1=Qty(value=t1_tc, min_val=1e-6, max_val=90e-6, unit="s"),
    # t2star=Qty(value=t2star_tc, min_val=1e-6, max_val=90e-6, unit="s"),
    # temp=Qty(value=init_temp, min_val=0.0, max_val=0.12, unit="K"),
)

q1q2 = Coupling(
    name="Q1-Q2",
    connected=["Qubit1", "Qubit2"],
    strength=Qty(
        value=coupling_strength, min_val=0 * 1e4, max_val=200e6, unit="Hz 2pi"
    ),
    hamiltonian_func=hamiltonians.int_XX,
)

model = Model(subsystems=[q1, q2], couplings=[q1q2], max_excitations=cut_excitations)
model.set_lindbladian(False)
model.set_dressed(True)
model.set_FR(True)

# ### MAKE GENERATOR
lo = devices.LO(name="lo", resolution=sim_res)
awg = devices.AWG(name="awg", resolution=awg_res)
dig_to_an = devices.DigitalToAnalog(name="dac", resolution=sim_res)
resp = devices.ResponseFFT(
    name="resp",
    rise_time=Qty(value=0.3e-9, min_val=0.05e-9, max_val=0.6e-9, unit="s"),
    resolution=sim_res,
)
mixer = devices.Mixer(name="mixer")

device_dict = {dev.name: dev for dev in [lo, awg, mixer, dig_to_an, resp]}
generator = Generator(
    devices=device_dict,
    chains={
        "Qubit1": {
            "lo": [],
            "awg": [],
            "dac": ["awg"],
            "resp": ["dac"],
            "mixer": ["lo", "resp"],
        },
        "Qubit2": {
            "lo": [],
            "awg": [],
            "dac": ["awg"],
            "resp": ["dac"],
            "mixer": ["lo", "resp"],
        },
    },
)

# ### MAKE GATESET
nodrive_env = Envelope(name="no_drive", params={}, shape=envelopes["no_drive"])
carrier_parameters = {
    "freq": Qty(value=0, min_val=0e9, max_val=10e9, unit="Hz 2pi"),
    "framechange": Qty(value=0.0, min_val=-3 * np.pi, max_val=5 * np.pi, unit="rad"),
}
carr_q1 = Carrier(
    name="carrier", desc="Frequency of the local oscillator", params=carrier_parameters
)
carr_q2 = copy.deepcopy(carr_q1)
carr_q2.params["freq"].set_value(freq_q2)

flux_params = {
    "amp": Qty(value=fluxamp, min_val=0.0, max_val=5, unit="V"),
    "t_final": Qty(
        value=cphase_time,
        min_val=0.5 * cphase_time,
        max_val=1.5 * cphase_time,
        unit="s",
    ),
    "t_up": Qty(
        value=0.2 * 1e-9, min_val=0.0 * cphase_time, max_val=0.5 * cphase_time, unit="s"
    ),
    "t_down": Qty(
        value=0.96 * cphase_time,
        min_val=0.5 * cphase_time,
        max_val=1.0 * cphase_time,
        unit="s",
    ),
    "risefall": Qty(
        value=0.2 * 1e-9, min_val=0.0 * cphase_time, max_val=1.0 * cphase_time, unit="s"
    ),
    "freq_offset": Qty(value=0, min_val=-50 * 1e6, max_val=50 * 1e6, unit="Hz 2pi"),
    "xy_angle": Qty(value=0, min_val=-0.5 * np.pi, max_val=2.5 * np.pi, unit="rad"),
}
flux_env = Envelope(
    name="flux",
    desc="Flux bias for tunable coupler",
    params=flux_params,
    shape=envelopes["flattop_cut"],
)

instr1 = Instruction(
    name="instr1",
    t_start=0.0,
    t_end=cphase_time,
    channels=["Qubit1"],
)

instr2 = Instruction(
    name="instr2",
    t_start=0.0,
    t_end=cphase_time,
    channels=["Qubit2"],
)
instr1.add_component(copy.deepcopy(flux_env), "Qubit1")
instr1.add_component(copy.deepcopy(carr_q1), "Qubit1")

instr2.add_component(flux_env, "Qubit2")
instr2.add_component(carr_q1, "Qubit2")

# ### MAKE EXPERIMENT
parameter_map = ParameterMap(
    instructions=[instr1, instr2], model=model, generator=generator
)
exp = Experiment(pmap=parameter_map, sim_res=sim_res)
exp.use_control_fields = False
exp.stop_partial_propagator_gradient = False

test_data = {}
with open("test/transmon_expanded.pickle", "rb") as filename:
    data = pickle.load(filename)

gen_signal1 = generator.generate_signals(instr1)
gen_signal2 = generator.generate_signals(instr2)


@pytest.mark.integration
def test_signals():
    test_data["signal_q1"] = gen_signal1["Qubit1"]
    test_data["signal_q2"] = gen_signal2["Qubit2"]
    np.testing.assert_allclose(
        actual=test_data["signal_q1"]["ts"],
        desired=data["signal_q1"]["ts"],
        atol=ATOL_SIG,
    )
    np.testing.assert_allclose(
        actual=test_data["signal_q1"]["values"],
        desired=data["signal_q1"]["values"],
        atol=ATOL_SIG,
    )
    np.testing.assert_allclose(
        actual=test_data["signal_q2"]["ts"],
        desired=data["signal_q2"]["ts"],
        atol=ATOL_SIG,
    )
    np.testing.assert_allclose(
        actual=test_data["signal_q2"]["values"].numpy(),
        desired=data["signal_q2"]["values"].numpy(),
        atol=ATOL_SIG,
    )


def test_chip_hamiltonians():
    vals = {"values": np.linspace(0, 1, 15)}
    test_data["fac_q1"] = q1.get_factor(phi_sig=vals["values"])
    test_data["prefac_q1"] = q1.get_prefactors(sig=vals["values"])
    test_data["q1_hams"] = q1.get_Hamiltonian(signal=vals)
    test_data["q2_hams"] = q2.get_Hamiltonian(signal=vals)

    np.testing.assert_allclose(test_data["fac_q1"], data["fac_q1"])
    for k in data["prefac_q1"]:
        np.testing.assert_allclose(test_data["prefac_q1"][k], data["prefac_q1"][k])
    np.testing.assert_allclose(actual=test_data["q1_hams"], desired=data["q1_hams"])
    np.testing.assert_allclose(actual=test_data["q2_hams"], desired=data["q2_hams"])


@pytest.mark.integration
def test_hamiltonians():
    model.max_excitations = 0
    test_data["hamiltonians_q1"] = model.get_Hamiltonian(gen_signal1)
    test_data["hamiltonians_q2"] = model.get_Hamiltonian(gen_signal2)

    np.testing.assert_allclose(
        actual=test_data["hamiltonians_q1"],
        desired=data["hamiltonians_q1"],
        atol=ATOL_FREQ,
    )

    np.testing.assert_allclose(
        actual=test_data["hamiltonians_q2"],
        desired=data["hamiltonians_q2"],
        atol=ATOL_FREQ,
    )


@pytest.mark.integration
def test_propagation():
    model.max_excitations = cut_excitations
    exp.set_opt_gates("instr1")
    exp.compute_propagators()
    test_data["propagators_q1"] = exp.propagators["instr1"]
    test_data["partial_propagators_q1"] = exp.partial_propagators["instr1"]
    exp.set_opt_gates("instr2")
    exp.compute_propagators()
    test_data["propagators_q2"] = exp.propagators["instr2"]
    test_data["partial_propagators_q2"] = exp.partial_propagators["instr2"]

    np.testing.assert_allclose(
        actual=test_data["propagators_q1"],
        desired=data["propagators_q1"],
        atol=ATOL_FREQ,
    )
    np.testing.assert_allclose(
        actual=test_data["partial_propagators_q1"],
        desired=data["partial_propagators_q1"],
        atol=ATOL_FREQ,
    )

    np.testing.assert_allclose(
        actual=test_data["propagators_q2"],
        desired=data["propagators_q2"],
        atol=ATOL_FREQ,
    )
    np.testing.assert_allclose(
        actual=test_data["partial_propagators_q2"],
        desired=data["partial_propagators_q2"],
        atol=ATOL_FREQ,
    )


@pytest.mark.unit
def test_save_and_load():
    exp.compute_propagators()
    propagators = exp.propagators
    cfg_str = hjson.dumpsJSON(exp.asdict(), default=hjson_encode)
    cfg_dct = hjson.loads(cfg_str, object_pairs_hook=hjson_decode)
    exp2 = Experiment(sim_res=sim_res)
    exp2.from_dict(cfg_dct)
    exp2.compute_propagators()
    for k in propagators:
        np.testing.assert_allclose(exp2.propagators[k], propagators[k])
