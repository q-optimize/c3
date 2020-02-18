"""Base run for c3 code."""

import os
import json
import pickle
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
datafile = cfg['datafile']
with open(datafile, 'rb+') as file:
    learn_from = pickle.load(file)

tf_utils.tf_setup()
with tf.device('/CPU:0'):
    exp = parsers.create_experiment(exp_setup, learn_from)
    opt = parsers.create_c3_opt(opt_config)
    opt.set_exp(exp)
    opt.read_data(datafile)
    dir = opt.logdir

    if 'initial_point' in cfg:
        init_point = cfg['initial_point']
        opt.load_best(init_point)
        os.system('cp {} {}/{}'.format(init_point, dir, 'init_point'))

    os.system('cp {} {}/{}'.format(__file__, dir, base(__file__)))
    os.system('cp {} {}/{}'.format(master_config, dir, base(master_config)))
    os.system('cp {} {}/{}'.format(exp_setup, dir, base(exp_setup)))
    os.system('cp {} {}/{}'.format(opt_config, dir, base(opt_config)))
    os.system('ln -s {} {}/{}'.format(datafile, dir, base(datafile)))

    opt.learn_model()
