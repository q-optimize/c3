"""Base run for c3 code."""
import os
import json
import argparse
import c3po.utils.parsers as parsers
import c3po.utils.tf_utils as tf_utils
import c3po.utils.utils as utils
import tensorflow as tf
from os.path import basename as base

parser = argparse.ArgumentParser()
parser.add_argument("master_config")
parser.add_argument("--double", nargs='?', const=True, default=False)
args = parser.parse_args()
master_config = args.master_config
with open(master_config, "r") as cfg_file:
    cfg = json.loads(cfg_file.read())
exp_setups = cfg['exp_setups']
c1_config = cfg['c1_config']
c2_config = cfg['c2_config']
c3_configs = cfg['c3_configs']
eval_func = cfg['eval_func']

tf_utils.tf_setup()
dir = utils.log_setup("/home/users/froy/c3loop/")
print(dir)
os.system('cp {} {}/{}'.format(__file__, dir, base(__file__)))
os.system('cp {} {}/{}'.format(master_config, dir, base(master_config)))
c1_dir = dir + 'c1/'
os.makedirs(c1_dir)
os.system('cp {} {}{}'.format(c1_config, c1_dir, base(c1_config)))
exp_setup = exp_setups[0]
os.system('cp {} {}{}'.format(exp_setup, c1_dir, base(exp_setup)))
if 'initial_point' in cfg:
    c1_init_point = cfg['initial_point']
    os.system('cp {} {}{}'.format(c1_init_point, c1_dir, 'init_point'))
c2_dir = dir + 'c2/'
os.makedirs(c2_dir)
os.system('cp {} {}{}'.format(c2_config, c2_dir, base(c2_config)))
os.system('cp {} {}{}'.format(eval_func, c2_dir, base(eval_func)))
c3_dirs = []
for indx in range(len(exp_setups)):
    c3_dir = dir + 'c3_' + str(indx) + '/'
    c3_dirs.append(c3_dir)
    os.makedirs(c3_dir)
    c3_config = c3_configs[indx]
    exp_setup = exp_setups[indx]
    os.system('cp {} {}{}'.format(c3_config, c3_dir, base(c3_config)))
    os.system('cp {} {}{}'.format(exp_setup, c3_dir, base(exp_setup)))
if args.double:
    c4_dir = dir + 'c4/'
    os.makedirs(c4_dir)
    os.system('cp {} {}{}'.format(c1_config, c4_dir, base(c1_config)))
    exp_setup = exp_setups[-1]
    os.system('cp {} {}{}'.format(exp_setup, c4_dir, base(exp_setup)))
    if 'initial_point' in cfg:
        c1_init_point = cfg['initial_point']
        os.system('cp {} {}{}'.format(c1_init_point, c4_dir, 'init_point'))
    c5_dir = dir + 'c5/'
    os.makedirs(c5_dir)
    os.system('cp {} {}{}'.format(c2_config, c5_dir, base(c2_config)))
    os.system('cp {} {}{}'.format(eval_func, c5_dir, base(eval_func)))

with tf.device('/CPU:0'):
    c1_opt = parsers.create_c1_opt(c1_dir + base(c1_config))
    exp = parsers.create_experiment(c1_dir + base(exp_setups[0]))
    c1_opt.set_exp(exp)
    c1_opt.replace_logdir(c1_dir)
    if 'initial_point' in cfg:
        c1_opt.load_best(c1_dir + 'init_point')
    c1_opt.optimize_controls()

    c2_opt, exp_right = parsers.create_c2_opt(
        c2_dir + base(c2_config),
        c2_dir + base(eval_func)
    )
    c2_opt.set_exp(exp)
    c2_opt.replace_logdir(c2_dir)
    c2_init_point = dir + 'c1/best_point_open_loop.log'
    c2_opt.load_best(c2_init_point)
    c2_opt.optimize_controls()

    for indx in range(len(exp_setups)):
        c3_dir = c3_dirs[indx]
        c3_config = c3_configs[indx]
        exp_setup = exp_setups[indx]
        c3_opt = parsers.create_c3_opt(c3_dir + base(c3_config))
        exp = parsers.create_experiment(c3_dir + base(exp_setup))
        c3_opt.set_exp(exp)
        c3_opt.replace_logdir(c3_dir)
        c3_datafile = dir + 'c2/learn_from.pickle'
        c3_opt.read_data(c3_datafile)
        if indx != 0:
            c3_init_point = c3_dirs[indx-1] + 'best_point_model_learn.log'
            c3_opt.load_best(c3_init_point)
        real = {'params': [
            par.numpy().tolist()
            for par in exp_right.get_parameters(c3_opt.opt_map)]
        }
        with open(c3_dir + "real_model_params.log", 'w') as real_file:
            real_file.write(json.dumps(c3_opt.opt_map))
            real_file.write("\n")
            real_file.write(json.dumps(real))
            real_file.write("\n")
            real_file.write(exp_right.print_parameters(c3_opt.opt_map))
        c3_opt.learn_model()

    if args.double:
        c4_opt = parsers.create_c1_opt(c4_dir + base(c1_config))
        exp = parsers.create_experiment(c4_dir + base(exp_setups[-1]))
        c4_opt.set_exp(exp)
        c4_opt.replace_logdir(c4_dir)
        if 'initial_point' in cfg:
            c4_opt.load_best(c4_dir + 'init_point')
        c4_opt.optimize_controls()

        c5_opt, exp_right = parsers.create_c2_opt(
            c5_dir + base(c2_config),
            c5_dir + base(eval_func)
        )
        c5_opt.set_exp(exp)
        c5_opt.replace_logdir(c5_dir)
        c5_init_point = dir + 'c4/best_point_open_loop.log'
        c5_opt.load_best(c5_init_point)
        c5_opt.optimize_controls()
