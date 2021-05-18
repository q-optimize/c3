import hjson

from c3.optimizers.c1 import C1
from c3.experiment import Experiment
from c3.main import run_cfg


def test_main_c1() -> None:
    with open("test/c1.cfg", "r") as cfg_file:
        cfg = hjson.load(cfg_file)
    run_cfg(cfg)


def test_create_c1() -> None:
    with open("test/c1.cfg", "r") as cfg_file:
        cfg = hjson.load(cfg_file)
    cfg.pop("optim_type")
    cfg.pop("gateset_opt_map")
    cfg.pop("opt_gates")

    exp = Experiment()
    exp.read_config(cfg.pop("exp_cfg"))
    C1(**cfg, pmap=exp.pmap)
