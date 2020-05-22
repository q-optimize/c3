#!/usr/bin/python -u
"""Base run for c3 code."""
import logging
logging.getLogger('tensorflow').disabled = True
import os
import shutil
import json
import pickle
import argparse
import c3po.utils.parsers as parsers
import c3po.utils.tf_utils as tf_utils
import tensorflow as tf

#import os
#import tensorflow as tf

#os.environ['AUTOGRAPH_VERBOSITY'] = 5
# Verbosity is now 5

#import os
#os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # or any {'0', '1', '2'}
#print(os.environ['TF_CPP_MIN_LOG_LEVEL'])
#import tensorflow as tf
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
#tf.get_logger().setLevel('ERROR')
# Verbosity is now 0


parser = argparse.ArgumentParser()
parser.add_argument("master_config")
args = parser.parse_args()

opt_config = args.master_config
with open(opt_config, "r") as cfg_file:
    try:
        cfg = json.loads(cfg_file.read())
    except json.decoder.JSONDecodeError:
        raise Exception(f"Config {opt_config} is invalid.")
optim_type = cfg['optim_type']
exp_setup = cfg['exp_setup']

tf_utils.tf_setup()
with tf.device('/CPU:0'):
    exp = parsers.create_experiment(exp_setup)

    if optim_type == "C1":
        opt = parsers.create_c1_opt(opt_config, exp.model.lindbladian)
    elif optim_type == "C2":
        eval_func = cfg['eval_func']
        opt, exp_right = parsers.create_c2_opt(opt_config, eval_func)
    elif optim_type == "C3":
        print("C3:STATUS: creating c3 opt ...")
        opt = parsers.create_c3_opt(opt_config)
    elif optim_type == "SET":
        print("C3:STATUS: creating set obj")
        opt = parsers.create_sensitivity_test(opt_config)
    elif optim_type == "confirm":
        print("C3:STATUS: creating c3 opt ...")
        opt = parsers.create_c3_opt(opt_config)
        opt.inverse = True
    else:
        raise Exception("C3:ERROR:Unknown optimization type specified.")
    opt.set_exp(exp)
    dir = opt.logdir

    shutil.copy2(exp_setup, dir)
    shutil.copy2(opt_config, dir)
    if optim_type == "C2":
        shutil.copy2(eval_func, dir)

    if 'initial_point' in cfg:
        initial_points = cfg['initial_point']
        if isinstance(initial_points, str):
            initial_points = [initial_points]
        elif isinstance(initial_points, list):
            pass
        else:
            raise Warning('initial_point has to be a path or a list of paths.')
        for init_point in initial_points:
            try:
                opt.load_best(init_point)
                print(
                    "C3:STATUS:Loading initial point from : "
                    f"{os.path.abspath(init_point)}"
                )
                init_dir = os.path.basename(os.path.normpath(os.path.dirname(init_point)))
                shutil.copy(init_point, dir + init_dir + "_initial_point.log")
            except FileNotFoundError:
                print(
                    f"C3:STATUS:No initial point found at "
                    f"{os.path.abspath(init_point)}. "
                    "Continuing with default."
                )

    if 'real_params' in cfg:
        real_params = cfg['real_params']

    if optim_type == "C1":
        if 'adjust_exp' in cfg:
            try:
                adjust_exp = cfg['adjust_exp']
                opt.adjust_exp(adjust_exp)
                print(
                    "C3:STATUS:Loading experimental values from : "
                    f"{os.path.abspath(adjust_exp)}"
                )
                shutil.copy(adjust_exp, dir+"adjust_exp.log")
            except FileNotFoundError:
                print(
                    f"C3:STATUS:No experimental values found at "
                    f"{os.path.abspath(adjust_exp)} "
                    "Continuing with default."
                )
        opt.optimize_controls()

    elif optim_type == "C2":
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

    elif optim_type == "C3" or optim_type == "confirm":
        learn_from = []
        opt.read_data(cfg['datafile'])
        key = list(cfg['datafile'].keys())[0]
        shutil.copy2(
            "/".join(cfg['datafile'][key].split("/")[0:-1]) \
            + "/real_model_params.log",
            dir
        )
        opt.learn_model()

    elif optim_type == "SET":
        learn_from = []
        opt.read_data(cfg['datafile'])
        shutil.copy2(
            "/".join(cfg['datafile']['left'].split("/")[0:-1]) \
            + "/real_model_params.log",
            dir
        )

        print("sensitivity test ...")
        opt.sensitivity_test()
