"""Base run for c3 code."""

import json
import pickle
import argparse
import c3po.utils.parsers as parsers
from c3po.utils.utils import replace_symlink as ln
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
        cp2(init_point, dir)

    cp2(__file__, dir)
    cp2(master_config, dir)
    cp2(exp_setup, dir)
    cp2(opt_config, dir)
    ln(datafile, dir+"model_learn.log")

    opt.learn_model()
