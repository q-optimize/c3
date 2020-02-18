"""Base run for c1 code."""

import os
import json
import argparse
import c3po.utils.parsers as parsers
import c3po.utils.tf_utils as tf_utils
import tensorflow as tf
from os.path import basename as base

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
    opt = parsers.create_c1_opt(opt_config)
    opt.set_exp(exp)
    dir = opt.logdir

    if 'initial_point' in cfg:
        init_point = cfg['initial_point']
        opt.load_best(init_point)
        os.system('cp {} {}/{}'.format(init_point, dir, 'init_point'))

    os.system('cp {} {}/{}'.format(__file__, dir, base(__file__)))
    os.system('cp {} {}/{}'.format(master_config, dir, base(master_config)))
    os.system('cp {} {}/{}'.format(exp_setup, dir, base(exp_setup)))
    os.system('cp {} {}/{}'.format(opt_config, dir, base(opt_config)))

    opt.optimize_controls()
