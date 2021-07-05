"""Integration Test module for Sensitivity Analysis on the dataset used for Model Learning
Please refer to test_model_learning.py for more details on the dataset
The test checks the integrity of the config file to ensure it creates the necessary optimizer
object which is then run. It also checks if the sweep goes till the higher end of the bounds, but
nothing else regarding the actual sweep run.
The integrity of the sweep algorithm is tested in test_scan_algos
"""

import hjson
import pytest
import numpy as np

from c3.optimizers.sensitivity import Sensitivity
from c3.experiment import Experiment

OPT_CONFIG_FILE_NAME = "test/sensitivity.cfg"
DESIRED_SWEEP_END_PARAMS = [-205000000.0, 5001500000.0]
SWEEP_PARAM_NAMES = ["Q1-anhar", "Q1-freq"]


@pytest.mark.integration
@pytest.mark.slow
def test_sensitivity() -> None:
    """Test sensitivity analysis with 1D sweeps on 2 variables"""
    with open(OPT_CONFIG_FILE_NAME, "r") as cfg_file:
        cfg = hjson.load(cfg_file)

    cfg.pop("optim_type")

    exp = Experiment()
    exp.read_config(cfg.pop("exp_cfg"))

    # test error handling for estimator
    with pytest.raises(NotImplementedError):
        opt = Sensitivity(**cfg, pmap=exp.pmap)

    cfg.pop("estimator")

    # test error handling for estimator_list
    with pytest.raises(NotImplementedError):
        opt = Sensitivity(**cfg, pmap=exp.pmap)

    cfg.pop("estimator_list")

    opt = Sensitivity(**cfg, pmap=exp.pmap)
    opt.set_exp(exp)
    opt.set_created_by(OPT_CONFIG_FILE_NAME)
    opt.run()

    for index, val in enumerate(SWEEP_PARAM_NAMES):
        # This decomposition of opt.sweep_end into the actual_param_end only makes
        # sense when you look at how opt.sweep_end structures the sweep endings
        actual_param_end = opt.sweep_end[index][val]["params"][0]
        np.testing.assert_almost_equal(
            actual_param_end, DESIRED_SWEEP_END_PARAMS[index]
        )
