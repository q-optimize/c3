#!/usr/bin/python -u

import argparse
import hjson
import numpy as np
from c3.utils.utils import num3str

parser = argparse.ArgumentParser()
parser.add_argument("log_file")
args = parser.parse_args()

with open(args.log_file) as file:
    log = hjson.load(file)

opt_map = log["opt_map"]
optim_status = log["optim_status"]
units = log["units"]
params = optim_status["params"]
grads = optim_status["gradient"]

print(f"Optimization reached {optim_status['goal']:0.3g} at {optim_status['time']}\n")
print("|------Parameter--------------------------------Value-----------Gradient--|")
ret = []
for ii in range(len(opt_map)):
    equiv_ids = opt_map[ii]
    par = params[ii]
    grad = grads[ii]

    par = num3str(par)
    if grad is None:
        grad = "Undefined "
    else:
        grad = num3str(grad)
    par_id = equiv_ids[0]
    nice_id = "-".join(par_id)
    ret.append(f"{nice_id:38}: {par+units[ii]:>16} {grad+units[ii]:>16}\n")
    if len(equiv_ids) > 1:
        for par_id in equiv_ids[1:]:
            ret.append("-".join(par_id))
            ret.append("\n")
        ret.append("\n")

print("".join(ret))
