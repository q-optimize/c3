"""
Test module to check if the evaluation number is increased in every step
"""

import hjson
import pytest
from c3.experiment import Experiment
from c3.optimizers.optimalcontrol import OptimalControl


@pytest.mark.slow
@pytest.mark.optimizers
# @pytest.mark.skip(reason="Data needs to be updated")
def test_c1_evaluation():
    with open("test/c1.cfg", "r") as cfg_file:
        cfg = hjson.load(cfg_file)
    cfg.pop("optim_type")
    cfg.pop("gateset_opt_map")
    cfg.pop("opt_gates")
    exp = Experiment(prop_method=cfg.pop("propagation_method", None))
    exp.read_config(cfg.pop("exp_cfg"))
    opt = OptimalControl(**cfg, pmap=exp.pmap)
    opt.set_exp(exp)
    opt.optimize_controls()
    assert (opt.evaluation == opt.options["maxfun"])
