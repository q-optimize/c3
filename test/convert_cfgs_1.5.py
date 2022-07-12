#!/usr/bin/env python3
import argparse
import hjson
import shutil

parser = argparse.ArgumentParser()
parser.add_argument(
    "files", metavar="F", type=str, nargs="+", help="List of files to convert"
)

args = parser.parse_args()


def convert(dct):
    if "Drives" not in dct:
        dct["Drives"] = {}
    drv_keys = []
    for key, coup in dct["Couplings"].items():
        if coup["c3type"] == "Drive":
            dct["Drives"][key] = coup
            drv_keys.append(key)
    for key in drv_keys:
        dct["Couplings"].pop(key, None)
    return dct


def write(cfg, dct):
    with open(cfg, "w") as cfg_file:
        hjson.dump(dct, cfg_file)


for cfg in args.files:
    with open(cfg, "r") as cfg_file:
        cfg_dct = hjson.load(cfg_file)
    if "model" in cfg_dct:
        if type(cfg_dct["model"]) is str:
            # Just convert separate model file
            model_str = cfg_dct["model"]
            shutil.copy(model_str, model_str + ".old")
            with open(model_str, "r") as cfg_file:
                dct = hjson.load(cfg_file)
            write(model_str, convert(dct))
        else:
            # Convert embedded model dict
            shutil.copy(cfg, cfg + ".old")
            dct = convert(cfg_dct["model"])
            cfg_dct["model"] = dct
            write(cfg, cfg_dct)

    elif "Couplings" in cfg_dct:
        # Convert model file
        shutil.copy(cfg, cfg + ".old")
        write(cfg, convert(cfg_dct))
