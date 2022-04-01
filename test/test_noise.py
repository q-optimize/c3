import os
import tempfile
import pytest
import pickle
import numpy as np

# Main C3 objects
from c3.optimizers.optimizer import TensorBoardLogger
from c3.experiment import Experiment as Exp

# Libs and helpers
import c3.libraries.algorithms as algorithms
import c3.libraries.fidelities as fidelities

from c3.optimizers.optimalcontrol_robust import OptimalControlRobust

logdir = os.path.join(tempfile.TemporaryDirectory().name, "c3logs")

gateset_opt_map = [
    [
        ("rx90p", "d1", "gauss", "amp"),
    ],
    [
        ("rx90p", "d1", "gauss", "freq_offset"),
    ],
    [
        ("rx90p", "d1", "gauss", "xy_angle"),
    ],
]


@pytest.mark.slow
@pytest.mark.optimizers
@pytest.mark.integration
# @pytest.mark.skip(reason="Data needs to be updated")
def test_c1_robust():
    exp = Exp()
    exp.read_config("test/noise_exp_1.hjson")
    exp.set_opt_gates(["rx90p"])
    pmap = exp.pmap
    pmap.set_opt_map(gateset_opt_map)
    noise_map = [[np.linspace(-0.1, 0.1, 5), [("dc_offset", "offset_amp")]]]
    opt = OptimalControlRobust(
        dir_path=logdir,
        fid_func=fidelities.unitary_infid_set,
        fid_subspace=["Q1"],
        pmap=pmap,
        noise_map=noise_map,
        algorithm=algorithms.lbfgs,
        options={"maxfun": 2},
        run_name="better_X90_tf_sgd",
        logger=[TensorBoardLogger()],
    )
    opt.set_exp(exp)

    opt.optimize_controls()

    assert opt.optim_status["goal"] < 0.1
    assert opt.current_best_goal < 0.1
    assert np.all(np.abs(opt.optim_status["gradient"]) > 0)
    assert np.all(np.abs(opt.optim_status["gradient_std"]) > 0)
    assert np.abs(opt.optim_status["goal_std"]) > 0

    with open("test/c1_robust.pickle", "rb") as f:
        data = pickle.load(f)

    data["c1_robust_lbfgs"] = opt.optim_status
    for k in ["goal", "goals_individual", "goal_std", "gradient", "gradient_std"]:
        desired = data["c1_robust_lbfgs"][k]
        np.testing.assert_allclose(opt.optim_status[k], desired)


@pytest.mark.slow
@pytest.mark.integration
def test_noise_devices():
    exp = Exp()
    exp.read_config("test/noise_exp_1.hjson")
    exp.set_opt_gates(["rx90p"])
    pmap = exp.pmap
    pmap.set_opt_map(gateset_opt_map)
    exp2 = Exp()
    exp2.read_config("test/noise_exp_2.hjson")
    exp.compute_propagators()
    fidelity0 = fidelities.average_infid_set(
        exp.propagators, pmap.instructions, index=[0], dims=exp.pmap.model.dims
    )

    noise_map = [
        [("pink_noise", "noise_amp")],
        [("dc_noise", "noise_amp")],
        [("awg_noise", "noise_amp")],
    ]
    for i in range(len(noise_map) + 1):
        params = np.zeros(len(noise_map))
        if i < len(noise_map):
            params[i] = 0.1

        exp2.pmap.set_parameters(params, noise_map)

        exp2.compute_propagators()
        fidelityA = fidelities.average_infid_set(
            exp2.propagators, pmap.instructions, index=[0], dims=exp.pmap.model.dims
        )
        pink_noiseA = exp2.pmap.generator.devices["PinkNoise"].signal["noise"]
        dc_noiseA = exp2.pmap.generator.devices["DCNoise"].signal["noise"]
        awg_noiseA = exp2.pmap.generator.devices["AWGNoise"].signal["noise-inphase"]

        exp2.compute_propagators()
        fidelityB = fidelities.average_infid_set(
            exp2.propagators, pmap.instructions, index=[0], dims=exp.pmap.model.dims
        )
        pink_noiseB = exp2.pmap.generator.devices["PinkNoise"].signal["noise"]
        dc_noiseB = exp2.pmap.generator.devices["DCNoise"].signal["noise"]
        awg_noiseB = exp2.pmap.generator.devices["AWGNoise"].signal["noise-inphase"]

        assert np.std(pink_noiseA) >= 0.05 * params[0]
        assert np.std(pink_noiseA) < 10 * params[0] + 1e-15
        if params[0] > 1e-15:
            assert np.median(np.abs(pink_noiseA - pink_noiseB) > 1e-10)

        if params[1] > 1e-15:
            assert np.abs(np.mean(dc_noiseA - dc_noiseB)) > 1e-6
            assert np.abs(np.mean(dc_noiseA - dc_noiseB)) < 10 * params[1]
        else:
            assert np.max(dc_noiseA - dc_noiseB) < 1e-15
        assert np.std(dc_noiseA) < 1e-15

        assert np.std(awg_noiseA) >= 0.05 * params[2]
        assert np.std(awg_noiseA) < 10 * params[2] + 1e-15
        if params[2] > 1e-15:
            assert np.mean(np.abs(awg_noiseA - awg_noiseB) > 1e-10)

        if np.max(params) > 0:
            assert fidelityA != fidelityB
            assert fidelity0 != fidelityB
        else:
            assert fidelityA == fidelityB
            assert fidelity0 == fidelityB
