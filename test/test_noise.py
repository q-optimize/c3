import pytest

import pickle
import numpy as np
import pytest

# Main C3 objects
from c3.c3objs import Quantity as Qty
from c3.parametermap import ParameterMap as Pmap
from c3.experiment import Experiment as Exp
from c3.system.model import Model as Mdl
from c3.generator.generator import Generator as Gnr

# Building blocks
import c3.generator.devices as devices
import c3.system.chip as chip
import c3.signal.pulse as pulse
import c3.signal.gates as gates

# Libs and helpers
import c3.libraries.algorithms as algorithms
import c3.libraries.hamiltonians as hamiltonians
import c3.libraries.fidelities as fidelities
import c3.libraries.envelopes as envelopes

from c3.optimizers.c1_robust import C1_robust

qubit_lvls = 3
freq_q1 = 5e9
anhar_q1 = -210e6
t1_q1 = 27e-6
t2star_q1 = 39e-6
qubit_temp = 50e-3

q1 = chip.Qubit(
    name="Q1",
    desc="Qubit 1",
    freq=Qty(value=freq_q1, min_val=4.995e9, max_val=5.005e9, unit="Hz 2pi"),
    anhar=Qty(value=anhar_q1, min_val=-380e6, max_val=-120e6, unit="Hz 2pi"),
    hilbert_dim=qubit_lvls,
    t1=Qty(value=t1_q1, min_val=1e-6, max_val=90e-6, unit="s"),
    t2star=Qty(value=t2star_q1, min_val=10e-6, max_val=90e-3, unit="s"),
    temp=Qty(value=qubit_temp, min_val=0.0, max_val=0.12, unit="K"),
)

drive = chip.Drive(
    name="d1",
    desc="Drive 1",
    comment="Drive line 1 on qubit 1",
    connected=["Q1"],
    hamiltonian_func=hamiltonians.x_drive,
)

model = Mdl(
    subsystems=[q1],
    couplings=[drive],
    tasks=[]
)

model.set_lindbladian(False)
model.set_dressed(True)

sim_res = 100e9  # Resolution for numerical simulation
awg_res = 2e9  # Realistic, limited resolution of an AWG

generator = Gnr(
    devices={
        "LO": devices.LO(name="lo", resolution=sim_res, outputs=1),
        "AWG": devices.AWG(name="awg", resolution=awg_res, outputs=1),
        "DigitalToAnalog": devices.DigitalToAnalog(
            name="dac", resolution=sim_res, inputs=1, outputs=1
        ),
        "Response": devices.Response(
            name="resp",
            rise_time=Qty(value=0.3e-9, min_val=0.05e-9, max_val=0.6e-9, unit="s"),
            resolution=sim_res,
            inputs=1,
            outputs=1,
        ),
        "Mixer": devices.Mixer(name="mixer", inputs=2, outputs=1),
        "DCOffset": devices.DC_Offset(
            name='dc_offset',
            offset_amp=Qty(value=0, min_val=-0.2, max_val=0.2, unit='V'),
            resolution=sim_res
        ),
        "VoltsToHertz": devices.VoltsToHertz(
            name="v_to_hz",
            V_to_Hz=Qty(value=1e9, min_val=0.9e9, max_val=1.1e9, unit="Hz/V"),
            inputs=1,
            outputs=1,
        ),
    },
    chains={
        "d1": ["LO", "AWG", "DigitalToAnalog", "Response", "Mixer", "DCOffset", "VoltsToHertz"],
    },
)

generator2 = Gnr(
    devices={
        "LO": devices.LO(name="lo", resolution=sim_res, outputs=1),
        "AWG": devices.AWG(name="awg", resolution=awg_res, outputs=1),
        "DigitalToAnalog": devices.DigitalToAnalog(
            name="dac", resolution=sim_res, inputs=1, outputs=1
        ),
        "Response": devices.Response(
            name="resp",
            rise_time=Qty(value=0.3e-9, min_val=0.05e-9, max_val=0.6e-9, unit="s"),
            resolution=sim_res,
            inputs=1,
            outputs=1,
        ),
        "Mixer": devices.Mixer(name="mixer", inputs=2, outputs=1),
        "DCOffset": devices.DC_Offset(
            name='dc_offset',
            offset_amp=Qty(value=0, min_val=-0.2, max_val=0.2, unit='V'),
            resolution=sim_res
        ),
        "Highpass": devices.HighpassFilter(
            name="highpass",
            cutoff=Qty(value=100e3 * 2 * np.pi, unit='Hz 2Pi'),
            rise_time=Qty(value=25e3 * 2 * np.pi, unit='Hz 2Pi'),
            resolution=sim_res
        ),
        "AWGNoise": devices.Additive_Noise(
            name='awg_noise',
            noise_amp=Qty(value=0, min_val=-0.01, max_val=1, unit='Phi0'),
            resolution=sim_res
        ),
        "DCNoise": devices.DC_Noise(
            name='dc_noise',
            noise_amp=Qty(value=0, min_val=0.00, max_val=1, unit='V'),
            resolution=sim_res
        ),
        "PinkNoise": devices.Pink_Noise(
            name='pink_noise',
            noise_strength=Qty(value=0, min_val=0.00, max_val=1, unit='V'),
            resolution=sim_res
        ),
        "VoltsToHertz": devices.VoltsToHertz(
            name="v_to_hz",
            V_to_Hz=Qty(value=1e9, min_val=0.9e9, max_val=1.1e9, unit="Hz/V"),
            inputs=1,
            outputs=1,
        ),
    },
    chains={
        "d1": ["LO", "AWG", "AWGNoise", "DigitalToAnalog", "Response", "Highpass", "Mixer", "DCNoise", "PinkNoise",
               "DCOffset", "VoltsToHertz"],
    },
)

t_final = 7e-9  # Time for single qubit gates
sideband = 50e6
gauss_params_single = {
    "amp": Qty(value=0.5, min_val=0.4, max_val=0.6, unit="V"),
    "t_final": Qty(
        value=t_final, min_val=0.5 * t_final, max_val=1.5 * t_final, unit="s"
    ),
    "sigma": Qty(value=t_final / 4, min_val=t_final / 8, max_val=t_final / 2, unit="s"),
    "xy_angle": Qty(value=0.0, min_val=-0.5 * np.pi, max_val=2.5 * np.pi, unit="rad"),
    "freq_offset": Qty(
        value=-sideband - 3e6, min_val=-56 * 1e6, max_val=-52 * 1e6, unit="Hz 2pi"
    ),
    "delta": Qty(value=-1, min_val=-5, max_val=3, unit=""),
}

gauss_env_single = pulse.Envelope(
    name="gauss",
    desc="Gaussian comp for single-qubit gates",
    params=gauss_params_single,
    shape=envelopes.gaussian_nonorm,
)

lo_freq = 5e9 + sideband
carrier_parameters = {
    "freq": Qty(value=lo_freq, min_val=4.5e9, max_val=6e9, unit="Hz 2pi"),
    "framechange": Qty(value=0.0, min_val=-np.pi, max_val=3 * np.pi, unit="rad"),
}
carr = pulse.Carrier(
    name="carrier", desc="Frequency of the local oscillator", params=carrier_parameters
)

X90p = gates.Instruction(name="X90p", t_start=0.0, t_end=t_final, channels=["d1"])

X90p.add_component(gauss_env_single, "d1")
X90p.add_component(carr, "d1")

pmap = Pmap([X90p], generator, model)

exp = Exp(pmap)

pmap2 = Pmap([X90p], generator2, model)
exp2 = Exp(pmap2)

exp.set_opt_gates(["X90p"])

gateset_opt_map = [
    [
        ("X90p", "d1", "gauss", "amp"),
    ],
    [
        ("X90p", "d1", "gauss", "freq_offset"),
    ],
    [
        ("X90p", "d1", "gauss", "xy_angle"),
    ]
]

pmap.set_opt_map(gateset_opt_map)

@pytest.mark.slow
@pytest.mark.optimizers
@pytest.mark.integration
def test_c1_robust():
    noise_map = [[np.linspace(-0.1, 0.1, 5), [('dc_offset', 'offset_amp')]]]
    opt = C1_robust(
        dir_path="/tmp/c3log/",
        fid_func=fidelities.average_infid_set,
        fid_subspace=["Q1"],
        pmap=pmap,
        noise_map=noise_map,
        algorithm=algorithms.lbfgs,
        options={"maxfun": 2},
        run_name="better_X90_tf_sgd",
    )

    opt.set_exp(exp)

    opt.optimize_controls()

    assert opt.optim_status['goal'] < 0.1
    assert opt.current_best_goal < 0.1
    assert np.all(np.abs(opt.optim_status['gradient']) > 0)
    assert np.all(np.abs(opt.optim_status['gradient_std']) > 0)
    assert np.abs(opt.optim_status['goal_std']) > 0

    with open('test/c1_robust.pickle', 'rb') as f:
        data = pickle.load(f)

    for k, v in data['c1_robust_lbfgs'].items():
        assert opt.optim_status[k] == v, f"{k} in optim_status not matching"


@pytest.mark.slow
@pytest.mark.integration
def test_noise_devices():
    exp.get_gates()
    fidelity0 = fidelities.average_infid_set(exp.unitaries, [1], exp.pmap.model.dims, 0, proj=True)

    noise_map = [[('pink_noise', 'noise_strength')], [('dc_noise', 'noise_amp')], [('awg_noise', 'noise_amp')]]
    for i in range(len(noise_map) + 1):
        params = np.zeros(len(noise_map))
        if i < len(noise_map):
            params[i] = 0.1

        exp2.pmap.set_parameters(params, noise_map)

        exp2.get_gates()
        fidelityA = fidelities.average_infid_set(exp2.unitaries, [1], exp2.pmap.model.dims, 0, proj=True)
        pink_noiseA = exp2.pmap.generator.devices["PinkNoise"].signal["noise"]
        dc_noiseA = exp2.pmap.generator.devices["DCNoise"].signal["noise"]
        awg_noiseA = exp2.pmap.generator.devices["AWGNoise"].signal["noise-inphase"]

        exp2.get_gates()
        fidelityB = fidelities.average_infid_set(exp2.unitaries, [1], exp2.pmap.model.dims, 0, proj=True)
        pink_noiseB = exp2.pmap.generator.devices["PinkNoise"].signal["noise"]
        dc_noiseB = exp2.pmap.generator.devices["DCNoise"].signal["noise"]
        awg_noiseB = exp2.pmap.generator.devices["AWGNoise"].signal["noise-inphase"]

        assert np.std(pink_noiseA) >= 0.5 * params[0]
        assert np.std(pink_noiseA) < 2 * params[0] + 1e-15
        if params[0] > 1e-15:
            assert np.median(np.abs(pink_noiseA - pink_noiseB) > 1e-10)

        if params[1] > 1e-15:
            assert np.abs(np.mean(dc_noiseA - dc_noiseB)) > 0.05 * params[1]
            assert np.abs(np.mean(dc_noiseA - dc_noiseB)) < 10 * params[1]
        else:
            assert np.max(dc_noiseA - dc_noiseB) < 1e-15
        assert np.std(dc_noiseA) < 1e-15

        assert np.std(awg_noiseA) >= 0.7 * params[2]
        assert np.std(awg_noiseA) < 1.3 * params[2] + 1e-15
        if params[2] > 1e-15:
            assert np.median(np.abs(awg_noiseA - awg_noiseB) > 1e-10)

        if np.max(params) > 0:
            assert fidelityA != fidelityB
            assert fidelity0 != fidelityB
        else:
            assert fidelityA == fidelityB
            assert fidelity0 == fidelityB
