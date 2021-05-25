#!/usr/bin/python3
"""Base script to run the C3 code from a main config file."""

import logging
import os
import hjson
import argparse
import c3.utils.tf_utils as tf_utils
import tensorflow as tf
from c3.parametermap import ParameterMap
from c3.experiment import Experiment
from c3.system.model import Model
from c3.generator.generator import Generator
from c3.optimizers.c1 import C1
from c3.optimizers.c2 import C2
from c3.optimizers.c3 import C3


logging.getLogger("tensorflow").disabled = True

# flake8: noqa: C901
def run_cfg(cfg, debug=False):
    """Execute an optimization problem described in the cfg file.

    Parameters
    ----------
    cfg : Dict[str, Union[str, int, float]]
        Configuration file containing optimization options and information needed to completely
        setup the system and optimization problem.
    debug : bool, optional
        Skip running the actual optimization, by default False

    """
    optim_type = cfg.pop("optim_type")
    optim_lib = {"C1": C1, "C2": C2, "C3": C3}
    if not optim_type in optim_lib:
        raise Exception("C3:ERROR:Unknown optimization type specified.")

    tf_utils.tf_setup()
    with tf.device("/CPU:0"):
        model = None
        gen = None
        if "model" in cfg:
            model = Model()
            model.read_config(cfg.pop("model"))
        if "generator" in cfg:
            gen = Generator()
            gen.read_config(cfg.pop("generator"))
        if "instructions" in cfg:
            pmap = ParameterMap(model=model, generator=gen)
            pmap.read_config(cfg.pop("instructions"))
            exp = Experiment(pmap)
        if "exp_cfg" in cfg:
            exp = Experiment()
            exp.read_config(cfg.pop("exp_cfg"))
        else:
            print("C3:STATUS: No instructions specified. Performing quick setup.")
            exp = Experiment()
            exp.quick_setup(cfg)

        exp.set_opt_gates(cfg.pop("opt_gates", None))
        gateset_opt_map = [
            [tuple(par) for par in pset] for pset in cfg.pop("gateset_opt_map")
        ]
        exp.pmap.set_opt_map(gateset_opt_map)

        opt = optim_lib[optim_type](**cfg, pmap=exp.pmap)
        opt.set_exp(exp)
        opt.set_created_by(cfg)

        if "initial_point" in cfg:
            initial_points = cfg["initial_point"]
            if isinstance(initial_points, str):
                initial_points = [initial_points]
            elif isinstance(initial_points, list):
                pass
            else:
                raise Warning("initial_point has to be a path or a list of paths.")
            for init_point in initial_points:
                try:
                    opt.load_best(init_point)
                    print(
                        "C3:STATUS:Loading initial point from : "
                        f"{os.path.abspath(init_point)}"
                    )
                    init_dir = os.path.basename(
                        os.path.normpath(os.path.dirname(init_point))
                    )
                except FileNotFoundError as fnfe:
                    raise Exception(
                        f"C3:ERROR:No initial point found at "
                        f"{os.path.abspath(init_point)}. "
                    ) from fnfe

        if "real_params" in cfg:
            real_params = cfg["real_params"]

        if optim_type == "C1":
            if "adjust_exp" in cfg:
                try:
                    adjust_exp = cfg["adjust_exp"]
                    opt.load_model_parameters(adjust_exp)
                    print(
                        "C3:STATUS:Loading experimental values from : "
                        f"{os.path.abspath(adjust_exp)}"
                    )
                except FileNotFoundError as fnfe:
                    raise Exception(
                        f"C3:ERROR:No experimental values found at "
                        f"{os.path.abspath(adjust_exp)} "
                        "Continuing with default."
                    ) from fnfe

        elif optim_type in ["C3", "confirm", "C3_confirm", "SET"]:
            opt.read_data(cfg["datafile"])

        if not debug:
            opt.run()


if __name__ == "__main__":
    os.nice(5)  # keep responsiveness when we overcommit memory

    tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

    parser = argparse.ArgumentParser()
    parser.add_argument("master_config")
    args = parser.parse_args()

    opt_config = args.master_config
    with open(opt_config, "r") as cfg_file:
        try:
            cfg = hjson.load(cfg_file)
        except hjson.decoder.HjsonDecodeError:
            raise Exception(f"Config {opt_config} is invalid.")
    run_cfg(cfg)
