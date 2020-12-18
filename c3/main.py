#!/usr/bin/python3
"""Base script to run the C3 code from a main config file."""

import logging
import os
import shutil
import hjson
import argparse
import c3.utils.parsers as parsers
import c3.utils.tf_utils as tf_utils
import tensorflow as tf
from c3.parametermap import ParameterMap
from c3.experiment import Experiment
from c3.system.model import Model
from c3.generator.generator import Generator

logging.getLogger("tensorflow").disabled = True

# flake8: noqa: C901
# TODO This script will be cleaned up
if __name__ == "__main__":

    os.nice(5)  # keep responsiveness when we overcommit memory

    tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

    parser = argparse.ArgumentParser()
    parser.add_argument("master_config")
    args = parser.parse_args()

    opt_config = args.master_config
    with open(opt_config, "r") as cfg_file:
        try:
            cfg = hjson.loads(cfg_file.read())
        except hjson.decoder.HjsonDecodeError:
            raise Exception(f"Config {opt_config} is invalid.")

    optim_type = cfg["optim_type"]

    tf_utils.tf_setup()
    with tf.device("/CPU:0"):
        model = None
        gen = None
        if "model" in cfg:
            model = Model()
            model.read_config(cfg["model"])
        if "generator" in cfg:
            gen = Generator()
            gen.read_config(cfg["generator"])
        pmap = ParameterMap(model=model, generator=gen)
        pmap.read_config(cfg["instructions"])
        exp = Experiment(pmap)

        if optim_type == "C1":
            opt = parsers.create_c1_opt(opt_config, exp)
            if cfg["include_model"]:
                opt.include_model()
        elif optim_type == "C2":
            eval_func = cfg["eval_func"]
            opt, exp_right = parsers.create_c2_opt(opt_config, eval_func)
        elif optim_type == "C3" or optim_type == "C3_confirm":
            print("C3:STATUS: creating c3 opt ...")
            opt = parsers.create_c3_opt(opt_config)
        elif optim_type == "SET":
            print("C3:STATUS: creating set obj")
            opt = parsers.create_sensitivity(opt_config)
        elif optim_type == "confirm":
            print("C3:STATUS: creating c3 opt ...")
            opt = parsers.create_c3_opt(opt_config)
            opt.inverse = True
        else:
            raise Exception("C3:ERROR:Unknown optimization type specified.")
        opt.set_exp(exp)
        opt.set_created_by(opt_config)

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
                except FileNotFoundError:
                    raise Exception(
                        f"C3:ERROR:No initial point found at "
                        f"{os.path.abspath(init_point)}. "
                    )

        if "real_params" in cfg:
            real_params = cfg["real_params"]

        if optim_type == "C1":
            if "adjust_exp" in cfg:
                try:
                    adjust_exp = cfg["adjust_exp"]
                    opt.load_best(adjust_exp)
                    opt.pmap.model.update_model()
                    print(
                        "C3:STATUS:Loading experimental values from : "
                        f"{os.path.abspath(adjust_exp)}"
                    )
                    shutil.copy(adjust_exp, dir + "adjust_exp.log")
                except FileNotFoundError:
                    raise Exception(
                        f"C3:ERROR:No experimental values found at "
                        f"{os.path.abspath(adjust_exp)} "
                        "Continuing with default."
                    )
            opt.optimize_controls()

        elif optim_type == "C2":
            real = {
                "params": [par.numpy().tolist() for par in exp_right.get_parameters()]
            }
            with open(dir + "real_model_params.log", "w") as real_file:
                real_file.write(hjson.dumps(exp_right.id_list))
                real_file.write("\n")
                real_file.write(hjson.dumps(real))
                real_file.write("\n")
                real_file.write(exp_right.print_parameters())

            opt.optimize_controls()

        elif optim_type == "C3" or optim_type == "confirm":
            learn_from = []
            opt.read_data(cfg["datafile"])
            key = list(cfg["datafile"].keys())[0]
            shutil.copy2(
                "/".join(cfg["datafile"][key].split("/")[0:-1])
                + "/real_model_params.log",
                dir,
            )
            opt.learn_model()

        elif optim_type == "C3_confirm":
            learn_from = []
            opt.read_data(cfg["datafile"])
            key = list(cfg["datafile"].keys())[0]
            shutil.copy2(
                "/".join(cfg["datafile"][key].split("/")[0:-1])
                + "/real_model_params.log",
                dir,
            )
            opt.confirm()

        elif optim_type == "SET":
            learn_from = []
            opt.read_data(cfg["datafile"])
            shutil.copy2(
                "/".join(list(cfg["datafile"].values())[0].split("/")[0:-1])
                + "/real_model_params.log",
                dir,
            )

            print("sensitivity test ...")
            opt.sensitivity()
