# System imports
import copy
import numpy as np
import pickle
import pytest
import hjson

# Main C3 objects
from c3.c3objs import Quantity as Qty, hjson_decode, hjson_encode
from c3.parametermap import ParameterMap as PMap
from c3.experiment import Experiment as Exp
from c3.model import Model as Mdl
from c3.generator.generator import Generator as Gnr

# Building blocks
import c3.generator.devices as devices
import c3.signal.gates as gates
import c3.libraries.chip as chip
import c3.signal.pulse as pulse

# Libs and helpers
import c3.libraries.envelopes as envelopes


EC = 330e6
EJ = 14.67e9

# define qubit
q1 = chip.CooperPairBox(
    name="Q1",
    desc="Qubit 1",
    EC=Qty(value=EC, min_val=0, max_val=400e9, unit="Hz 2pi"),
    EJ=Qty(value=EJ, min_val=0, max_val=30e9, unit="Hz 2pi"),
    hilbert_dim=3,
    NG=Qty(value=0, min_val=-5, max_val=5, unit=""),
    Asym=Qty(value=0, min_val=-1, max_val=1, unit=""),
    Reduced_Flux=Qty(value=0, min_val=-1, max_val=1, unit=""),
    calc_dim=Qty(value=21, min_val=1, max_val=41, unit=""),
)

model = Mdl([q1], [], [])
model.use_FR = False
model.lindbladian = False
model.dressed = False

# define control signals
sim_res = 30e9
awg_res = 10e9
lo = devices.LO(name="lo", resolution=sim_res)
awg = devices.AWG(name="awg", resolution=awg_res)
mixer = devices.Mixer(name="mixer")
dig_to_an = devices.DigitalToAnalog(name="dac", resolution=sim_res)

generator = Gnr(
    devices={
        "LO": devices.LO(name="lo", resolution=sim_res, outputs=1),
        "AWG": devices.AWG(name="awg", resolution=awg_res, outputs=1),
        "DigitalToAnalog": devices.DigitalToAnalog(
            name="dac", resolution=sim_res, inputs=1, outputs=1
        ),
        "Mixer": devices.Mixer(name="mixer", inputs=2, outputs=1),
        "DC-Offset": devices.DC_Offset(name="DC-Offset", inputs=1, outputs=1),
    },
    chains={
        "Q1": {
            "LO": [],
            "AWG": [],
            "DigitalToAnalog": ["AWG"],
            "Mixer": ["LO", "DigitalToAnalog"],
            "DC-Offset": ["Mixer"],
        },
    },
)

carrier_parameters = {
    "freq": Qty(
        value=5381.790179180943e6,
        min_val=4.5e9,
        max_val=7.5e9,
        unit="Hz 2pi",
    ),
    "framechange": Qty(value=0, min_val=-np.pi, max_val=3 * np.pi, unit="rad"),
}
carr = pulse.Carrier(
    name="carrier", desc="Frequency of the local oscillator", params=carrier_parameters
)


t_final = 100e-9
gauss_params_single = {
    "amp": Qty(value=0.00149758282986231 / 2, min_val=-0.01, max_val=0.01, unit="V"),
    "t_final": Qty(
        value=t_final, min_val=0.5 * t_final, max_val=1.5 * t_final, unit="s"
    ),
    "sigma": Qty(value=t_final / 4, min_val=t_final / 8, max_val=t_final / 2, unit="s"),
    "xy_angle": Qty(value=0.04, min_val=-0.5 * np.pi, max_val=2.5 * np.pi, unit="rad"),
    "freq_offset": Qty(value=-1e1, min_val=-50 * 1e6, max_val=50 * 1e6, unit="Hz 2pi"),
    "delta": Qty(
        value=480, min_val=-1.2e5, max_val=1.2e5, unit=""  # ,10,#10e9/(g12-f12),
    ),
    "offset_amp": Qty(value=0, min_val=0, max_val=1, unit="V"),
    "t_rise": Qty(value=1e-9, min_val=0, max_val=t_final / 2.0, unit="s"),
    "t_bin_start": Qty(value=0.0, min_val=0, max_val=1, unit=""),
    "t_bin_end": Qty(value=1499, min_val=-1, max_val=1e9, unit=""),
    "inphase": Qty(value=0, min_val=-1, max_val=1, unit=""),
}


gauss_env_single = pulse.Envelope(
    name="gauss",
    desc="Gaussian comp for single-qubit gates",
    params=gauss_params_single,
    shape=envelopes.gaussian_nonorm,
)


instr1 = gates.Instruction(
    name="instr1",
    targets=[0],
    t_start=0.0,
    t_end=t_final,
    channels=["Q1"],
    ideal=np.array([[1, 0, 0], [0, 0, 1], [0, 1, 0]]),
)
instr1.add_component(copy.deepcopy(gauss_env_single), "Q1")
instr1.add_component(carr, "Q1")

instr2 = gates.Instruction(
    name="instr2",
    targets=[0],
    t_start=0.0,
    t_end=t_final,
    channels=["Q1"],
    ideal=np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]]),
)
instr2.add_component(copy.deepcopy(gauss_env_single), "Q1")
instr2.add_component(copy.deepcopy(carr), "Q1")

instr2.comps["Q1"]["gauss"].params["offset_amp"].set_value(0.5)

# Make Experiment
parameter_map = PMap(instructions=[instr1, instr2], model=model, generator=generator)
exp = Exp(pmap=parameter_map, sim_res=sim_res)
exp.use_control_fields = False
exp.stop_partial_propagator_gradient = False

gen_signal1 = generator.generate_signals(instr1)
gen_signal2 = generator.generate_signals(instr2)

test_data = {}

with open("test/CooperPairBox.pickle", "rb") as filename:
    data = pickle.load(filename)


@pytest.mark.integration
def test_signals():
    test_data["instr1"] = gen_signal1["Q1"]
    test_data["instr2"] = gen_signal2["Q1"]
    np.testing.assert_allclose(
        actual=test_data["instr1"]["ts"],
        desired=data["instr1"]["ts"],
        atol=1e-11 * np.max(data["instr1"]["ts"]),
    )
    np.testing.assert_allclose(
        actual=test_data["instr1"]["values"],
        desired=data["instr1"]["values"],
        atol=1e-11 * np.max(data["instr1"]["values"]),
    )
    np.testing.assert_allclose(
        actual=test_data["instr2"]["ts"],
        desired=data["instr2"]["ts"],
        atol=1e-11 * np.max(data["instr2"]["ts"]),
    )
    np.testing.assert_allclose(
        actual=test_data["instr2"]["values"].numpy(),
        desired=data["instr2"]["values"].numpy(),
        atol=1e-11 * np.max(data["instr"]["values"]),
    )


@pytest.mark.integration
def test_static_hamiltonian():
    test_data["static_h"] = model.get_Hamiltonian()
    np.testing.assert_allclose(actual=test_data["static_h"], desired=data["static_h"])


@pytest.mark.integration
def test_hamiltonians():
    test_data["hamiltonians_instr1"] = model.get_Hamiltonian(gen_signal1)
    test_data["hamiltonians_instr2"] = model.get_Hamiltonian(gen_signal2)

    np.testing.assert_allclose(
        actual=test_data["hamiltonians_instr1"],
        desired=data["hamiltonians_instr1"],
        atol=1e-9 * np.max(data["hamiltonians_instr1"]),
    )

    np.testing.assert_allclose(
        actual=test_data["hamiltonians_instr2"],
        desired=data["hamiltonians_instr2"],
        atol=1e-11 * np.max(data["hamiltonians_instr2"]),
    )


@pytest.mark.integration
def test_propagation():
    exp.set_opt_gates("instr1")
    exp.compute_propagators()
    test_data["propagators_1"] = exp.propagators["instr1"]
    test_data["partial_propagators_1"] = exp.partial_propagators["instr1"]
    exp.set_opt_gates("instr2")
    exp.compute_propagators()
    test_data["propagators_2"] = exp.propagators["instr2"]
    test_data["partial_propagators_2"] = exp.partial_propagators["instr2"]

    np.testing.assert_allclose(
        actual=test_data["propagators_1"],
        desired=data["propagators_1"],
        atol=1e-11 * np.max(data["propagators_1"]),
    )
    np.testing.assert_allclose(
        actual=test_data["partial_propagators_1"],
        desired=data["partial_propagators_1"],
        atol=1e-11 * np.max(data["partial_propagators_1"]),
    )

    np.testing.assert_allclose(
        actual=test_data["propagators_2"],
        desired=data["propagators_2"],
        atol=1e-11 * np.max(data["propagators_2"]),
    )
    np.testing.assert_allclose(
        actual=test_data["partial_propagators_2"],
        desired=data["partial_propagators_2"],
        atol=1e-11 * np.max(data["partial_propagators_2"]),
    )


@pytest.mark.unit
def test_save_and_load():
    exp.compute_propagators()
    propagators = exp.propagators
    cfg_str = hjson.dumpsJSON(exp.asdict(), default=hjson_encode)
    cfg_dct = hjson.loads(cfg_str, object_pairs_hook=hjson_decode)
    exp2 = Exp(sim_res=sim_res)
    exp2.from_dict(cfg_dct)
    exp2.compute_propagators()
    for k in propagators:
        np.testing.assert_allclose(exp2.propagators[k], propagators[k])
