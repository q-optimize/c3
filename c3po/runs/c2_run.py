"""Base run for c2 code."""

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
eval_func = cfg['eval_func']

tf_utils.tf_setup()
with tf.device('/CPU:0'):
    exp = parsers.create_experiment(exp_setup)
    opt, exp_right = parsers.create_c2_opt(opt_config, eval_func)
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
    cp2(eval_func, dir)
    real = {'params': [
        par.numpy().tolist()
        for par in exp_right.get_parameters()]
    }
    with open(dir + "real_model_params.log", 'w') as real_file:
        real_file.write(json.dumps(exp_right.id_list))
        real_file.write("\n")
        real_file.write(json.dumps(real))
        real_file.write("\n")
        real_file.write(exp_right.print_parameters())

    opt.optimize_controls()
