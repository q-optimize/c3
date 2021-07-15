"""Test module to check if optimizer classes are initialized correcty by the main file.
"""

import hjson
import pytest

from c3.optimizers.optimalcontrol import OptimalControl
from c3.optimizers.modellearning import ModelLearning
from c3.experiment import Experiment
from c3.main import run_cfg


@pytest.mark.integration
def test_main_c1() -> None:
    with open("test/c1.cfg", "r") as cfg_file:
        cfg = hjson.load(cfg_file)
    run_cfg(cfg, "test/c1.cfg", debug=True)


@pytest.mark.integration
def test_main_c2() -> None:
    with open("test/c2.cfg", "r") as cfg_file:
        cfg = hjson.load(cfg_file)
    run_cfg(cfg, "test/c2.cfg", debug=True)


@pytest.mark.integration
def test_main_c3() -> None:
    with open("test/c3.cfg", "r") as cfg_file:
        cfg = hjson.load(cfg_file)
    run_cfg(cfg, "test/c3.cfg", debug=True)


@pytest.mark.integration
def test_main_sens() -> None:
    with open("test/sensitivity.cfg", "r") as cfg_file:
        cfg = hjson.load(cfg_file)

    # pop configs that are not yet supported
    cfg.pop("estimator")
    cfg.pop("estimator_list")
    run_cfg(cfg, "test/sensitivity.cfg", debug=True)


@pytest.mark.integration
def test_create_c1() -> None:
    with open("test/c1.cfg", "r") as cfg_file:
        cfg = hjson.load(cfg_file)
    cfg.pop("optim_type")
    cfg.pop("gateset_opt_map")
    cfg.pop("opt_gates")

    exp = Experiment()
    exp.read_config(cfg.pop("exp_cfg"))
    OptimalControl(**cfg, pmap=exp.pmap)


@pytest.mark.integration
def test_create_c3() -> None:
    with open("test/c3.cfg", "r") as cfg_file:
        cfg = hjson.load(cfg_file)
    cfg.pop("optim_type")
    cfg.pop("exp_opt_map")

    exp = Experiment()
    exp.read_config(cfg.pop("exp_cfg"))
    assert isinstance(ModelLearning(**cfg, pmap=exp.pmap), ModelLearning)
