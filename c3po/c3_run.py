"""Base run for c3 code"""

import os
import argparse
import c3po.parsers
from c3po.experiment import Experiment as Exp

parser = argparse.ArgumentParser()
parser.add_argument("-model")
parser.add_argument("-generator")
parser.add_argument("-gateset")
parser.add_argument("-optimizer")
parser.add_argument("-datafile")
args = parser.parse_args()

model = c3po.parsers.create_model(args.model)
generator = c3po.parsers.create_generator(args.generator)
gates = c3po.parsers.create_gates(args.gateset)
exp = Exp(model=model, generator=generator, gates=gates)
opt = c3po.parsers.create_optimizer(args.optimizer)
opt.read_data(args.datafile)
opt.set_exp(exp)
logdir = opt.log_setup("/localdisk/c3logs/")

os.system('cp {} {}/{}'.format(__file__, logdir, os.path.basename(__file__)))
os.system('cp {} {}/{}'.format(args.model, logdir, args.model))
os.system('cp {} {}/{}'.format(args.generator, logdir, args.generator))
os.system('cp {} {}/{}'.format(args.gateset, logdir, args.gateset))
os.system('cp {} {}/{}'.format(args.optimizer, logdir, args.optimizer))

opt.learn_model()
