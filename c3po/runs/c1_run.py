"""Base run for c1 code."""

import os
import json
import argparse
import c3po.utils.parsers as parsers
import c3po.utils.tf_utils as tf_utils
import tensorflow as tf
from shutil import copy2 as cp2

parser = argparse.ArgumentParser()
parser.add_argument("master_config")
args = parser.parse_args()
master_config = args.master_config
with open(master_config, "r") as cfg_file:
    cfg = json.loads(cfg_file.read())
exp_setup = cfg['exp_setup']
opt_config = cfg['optimizer_config']


tf_utils.tf_setup()
with tf.device('/CPU:0'):
    exp = parsers.create_experiment(exp_setup)

    if 'adjust_exp' in cfg:
        adjust_exp = cfg['adjust_exp']
        with open(adjust_exp) as file:
            best = file.readlines()
            best_exp_opt_map = [tuple(a) for a in json.loads(best[0])]
            p = json.loads(best[1])['params']
            exp.set_parameters(p, best_exp_opt_map)
            print("\nLoading best experiment point.")

    opt = parsers.create_c1_opt(opt_config)
    opt.set_exp(exp)
    dir = opt.logdir

    if 'initial_point' in cfg:
        init_point = cfg['initial_point']
        opt.load_best(init_point)
        cp2(init_point, dir)

    cp2(__file__, dir)
    cp2(master_config, dir)
    cp2(exp_setup, dir)
    cp2(opt_config, dir)

    opt.optimize_controls()
