import copy
import pickle

import hjson
from numpy.testing import assert_array_almost_equal as almost_equal

from c3.libraries.fidelities import state_transfer_infid_set
from c3.signal.gates import Instruction

from c3.c3objs import Quantity, hjson_decode, hjson_encode
from c3.experiment import Experiment
from c3.generator.generator import Generator
from c3.libraries.envelopes import envelopes
from c3.parametermap import ParameterMap
from c3.signal import gates, pulse
from c3.model import Model
import numpy as np
import pytest
from c3.libraries.constants import GATES


model = Model()
model.read_config("test/test_model.cfg")
generator = Generator()
generator.read_config("test/generator2.cfg")

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

gauss_env_single = pulse.EnvelopeDrag(
    name="gauss",
    desc="Gaussian comp for single-qubit gates",
    params=gauss_params_single,
    shape=envelopes["gaussian_nonorm"],
)

lo_freq_q1 = 5e9 + sideband
carrier_parameters = {
    "freq": Quantity(value=lo_freq_q1, min_val=4.5e9, max_val=6e9, unit="Hz 2pi"),
    "framechange": Quantity(value=0.0, min_val=-np.pi, max_val=3 * np.pi, unit="rad"),
}

carr = pulse.Carrier(
    name="carrier", desc="Frequency of the local oscillator", params=carrier_parameters
)

lo_freq_q2 = 5.6e9 + sideband
carr_2 = copy.deepcopy(carr)
carr_2.params["freq"].set_value(lo_freq_q2)

instr = gates.Instruction(
    name="multi_instruction",
    t_start=0.0,
    t_end=t_final * 3 + 6e-9,
    channels=["d1", "d2"],
)

instr.add_component(copy.deepcopy(gauss_env_single), "d1", name="gaussd1_1")
instr.add_component(
    copy.deepcopy(gauss_env_single),
    "d1",
    name="gaussd1_2",
    options={"delay": Quantity(1e-9), "trigger_comp": ("d1", "gaussd1_1")},
)
instr.add_component(
    copy.deepcopy(gauss_env_single),
    "d1",
    name="gaussd1_3",
    options={"delay": Quantity(1e-9), "trigger_comp": ("d1", "gaussd1_2")},
)
instr.add_component(copy.deepcopy(gauss_env_single), "d2", name="gaussd2_1")
instr.add_component(
    copy.deepcopy(gauss_env_single),
    "d2",
    name="gaussd2_2",
    options={
        "delay": Quantity(1e-9),
        "trigger_comp": ("d1", "gaussd1_2"),
        "t_final_cut": Quantity(0.9 * t_final),
    },
)
instr.add_component(carr, "d1")
instr.add_component(carr_2, "d2")

instr_dict_str = hjson.dumpsJSON(instr.asdict(), default=hjson_encode)
pmap = ParameterMap(model=model, generator=generator, instructions=[instr])

exp = Experiment(pmap)

with open("test/instruction.pickle", "rb") as filename:
    test_data = pickle.load(filename)


@pytest.mark.integration
def test_extended_pulse():
    instr_it = instr
    gen_signal = generator.generate_signals(instr_it)
    ts = gen_signal["d1"]["ts"]

    np.testing.assert_allclose(
        ts,
        test_data["signal"]["d1"]["ts"],
        atol=1e-9 * np.max(test_data["signal"]["d1"]["ts"]),
    )
    np.testing.assert_allclose(
        actual=gen_signal["d1"]["values"].numpy(),
        desired=test_data["signal"]["d1"]["values"].numpy(),
        atol=1e-9 * np.max(test_data["signal"]["d1"]["values"].numpy()),
    )
    np.testing.assert_allclose(
        actual=gen_signal["d2"]["values"].numpy(),
        desired=test_data["signal"]["d2"]["values"].numpy(),
        atol=1e-9 * np.max(test_data["signal"]["d2"]["values"].numpy()),
    )
    np.testing.assert_allclose(
        instr_it.get_full_gate_length(), test_data["full_gate_length1"]
    )
    instr_it.auto_adjust_t_end(buffer=0.2)
    np.testing.assert_allclose(
        instr_it.get_full_gate_length(), test_data["full_gate_length2"]
    )
    np.testing.assert_allclose(instr_it.t_end, test_data["t_end2"])

    pmap.set_parameters(
        [2 * t_final],
        [[("multi_instruction", "d1", "gaussd1_2", "t_final")]],
        extend_bounds=True,
    )

    gen_signal = generator.generate_signals(instr_it)
    np.testing.assert_allclose(
        actual=gen_signal["d1"]["values"].numpy(),
        desired=test_data["signal2"]["d1"]["values"].numpy(),
        atol=1e-9 * np.max(test_data["signal"]["d1"]["values"].numpy()),
    )
    np.testing.assert_allclose(
        actual=gen_signal["d2"]["values"].numpy(),
        desired=test_data["signal2"]["d2"]["values"].numpy(),
        atol=1e-9 * np.max(test_data["signal"]["d2"]["values"].numpy()),
    )
    instr_it.auto_adjust_t_end(0.1)
    gen_signal = generator.generate_signals(instr_it)
    np.testing.assert_allclose(
        actual=gen_signal["d1"]["values"].numpy(),
        desired=test_data["signal3"]["d1"]["values"].numpy(),
        atol=1e-9 * np.max(test_data["signal"]["d1"]["values"].numpy()),
    )
    np.testing.assert_allclose(
        actual=gen_signal["d2"]["values"].numpy(),
        desired=test_data["signal3"]["d2"]["values"].numpy(),
        atol=1e-9 * np.max(test_data["signal"]["d2"]["values"].numpy()),
    )


@pytest.mark.unit
def test_save_and_load():
    global instr, pmap
    instr = Instruction()
    instr.from_dict(hjson.loads(instr_dict_str, object_pairs_hook=hjson_decode))
    pmap = ParameterMap(model=model, generator=generator, instructions=[instr])
    test_extended_pulse()


@pytest.mark.unit
def test_str_conversion():
    assert repr(instr) == "Instruction[multi_instruction]"


@pytest.mark.unit
def test_set_name_ideal():
    """Check that asigning a name of a specific gate from the constants updates the ideal unitary."""
    instr.set_name("ry90p")
    assert (instr.ideal == GATES["ry90p"]).all()
    instr.set_name("crzp")
    assert (instr.ideal == GATES["crzp"]).all()


@pytest.mark.unit
def test_correct_ideal_assignment() -> None:
    custom_gate = np.array([[1, 0, 0, 0],
                            [0, 1, 0, 0],
                            [0, 0, 0, 1],
                            [0, 0, 1, 0]], dtype=np.complex)
    propagators = {"custom": custom_gate}
    instructions = {"custom": Instruction("custom", ideal=custom_gate)}
    psi_0 = np.array([[1], [0], [0], [0]])
    goal = state_transfer_infid_set(
        propagators=propagators,
        instructions=instructions,
        index=[0, 1],
        dims=[2, 2],
        psi_0=psi_0,
        n_eval=136,
    )
    almost_equal(goal, 0)