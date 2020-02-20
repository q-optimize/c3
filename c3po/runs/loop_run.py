"""Base run for c3 code."""
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
c1_config = cfg['c1_config']
c2_config = cfg['c2_config']
c3_config = cfg['c3_config']
eval_func = cfg['eval_func']



tf_utils.tf_setup()
with tf.device('/CPU:0'):
    exp = parsers.create_experiment(exp_setup)

    c1_opt = parsers.create_c1_opt(c1_config)
    c1_opt.set_exp(exp)
    dir = c1_opt.logdir

    os.system('cp {} {}/{}'.format(__file__, dir, base(__file__)))
    os.system('cp {} {}/{}'.format(master_config, dir, base(master_config)))
    os.system('cp {} {}/{}'.format(exp_setup, dir, base(exp_setup)))
    os.system('cp {} {}/{}'.format(c1_config, dir, base(c1_config)))
    os.system('cp {} {}/{}'.format(c2_config, dir, base(c2_config)))
    os.system('cp {} {}/{}'.format(c3_config, dir, base(c3_config)))
    os.system('cp {} {}/{}'.format(eval_func, dir, base(eval_func)))

    if 'initial_point' in cfg:
        c1_init_point = cfg['initial_point']
        c1_opt.load_best(c1_init_point)
        os.system('cp {} {}/{}'.format(c1_init_point, dir, 'init_point'))
    c1_opt.optimize_controls()

    c2_opt = parsers.create_c2_opt(c2_config, eval_func)
    c2_opt.set_exp(exp)
    c2_opt.replace_logdir(dir)
    c2_init_point = dir + 'best_point_open_loop.log'
    c2_opt.load_best(c2_init_point)
    c2_opt.optimize_controls()

    c3_opt = parsers.create_c3_opt(c3_config)
    c3_opt.set_exp(exp)
    c3_opt.replace_logdir(dir)
    c3_datafile = dir + 'learn_from.pickle'
    c3_opt.read_data(c3_datafile)
    c3_opt.learn_model()
