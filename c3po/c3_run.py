"""Base run for c3 code."""

import os
import json
import argparse
import c3po.parsers

parser = argparse.ArgumentParser()
parser.add_argument("master_config")
args = parser.parse_args()
master_config = args.master_config
with open(master_config, "r") as cfg_file:
    cfg = json.loads(cfg_file.read())
experiment_setup = cfg['experiment_setup']
optimizer_config = cfg['optimizer_config']
datafile = cfg['datafile']

exp = c3po.parsers.creat_experiment(experiment_setup, datafile)
opt = c3po.parsers.create_optimizer(optimizer_config)
opt.set_exp(exp)
opt.read_data(datafile)
logdir = opt.log_setup()

os.system('cp {} {}/{}'.format(__file__, logdir, os.path.basename(__file__)))
os.system('cp {} {}/{}'.format(master_config, logdir, master_config))
os.system('cp {} {}/{}'.format(experiment_setup, logdir, experiment_setup))
os.system('cp {} {}/{}'.format(optimizer_config, logdir, optimizer_config))
os.system('ln -s {} {}/{}'.format(datafile, logdir, datafile))

opt.learn_model()
