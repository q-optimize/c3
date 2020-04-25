import os
import json
import time
import random
import numpy as np
import matplotlib.pyplot as plt
from c3po.libraries.algorithms import algorithms
from c3po.libraries.estimators import estimators
from c3po.libraries.fidelities import fidelities
from c3po.libraries.sampling import sampling
from c3po.utils.display import plots
import c3po.utils.qt_utils as qt_utils
from runpy import run_path
from c3po.optimizers.c1 import C1
from c3po.optimizers.c2 import C2
from c3po.optimizers.c3 import C3
from c3po.tasks.sensitivity import SET


def create_experiment(exp_setup, datafile=''):
    exp_namespace = run_path(exp_setup)
    if datafile:
        exp = exp_namespace['create_experiment'](datafile)
    else:
        exp = exp_namespace['create_experiment']()
    return exp


def create_c1_opt(optimizer_config, lindblad):
    with open(optimizer_config, "r") as cfg_file:
        cfg = json.loads(cfg_file.read())

    if lindblad:
        fid = 'lindbladian_' + cfg['fid_func']
    else:
        fid = cfg['fid_func']

    if lindblad:
        cb_fids = ['lindbladian_' + f for f in cfg['callback_fids']]
    else:
        cb_fids = cfg['callback_fids']

    try:
        fid_func = fidelities[fid]
    except KeyError:
        raise Exception(
            f"C3:ERROR:Unkown goal function: {fid} "
        )
    print(f"C3:STATUS:Found {fid} in libraries.")
    callback_fids = []
    for cb_fid in cb_fids:
        try:
            cb_fid_func = fidelities[cb_fid]
        except KeyError:
            raise Exception(
                f"C3:ERROR:Unkown goal function: {cb_fid}"
            )
        print(f"C3:STATUS:Found {cb_fid} in libraries.")
        callback_fids.append(cb_fid_func)
    opt_gates = cfg['opt_gates']
    gateset_opt_map = [
        [tuple(par) for par in set]
        for set in cfg['gateset_opt_map']
    ]
    algorithm = algorithms[cfg['algorithm']]
    options = {}
    if 'options' in cfg:
        options = cfg['options']
    if 'plot_dynamics' in cfg:
        if cfg['plot_dynamics'] == "False":
            plot_dynamics = False
        elif cfg['plot_dynamics'] == "True":
            plot_dynamics = True
        else:
            raise(Exception("Couldn't resolve setting of 'plot_dynamics'"))
    else:
        plot_dynamics = False
    if 'plot_pulses' in cfg:
        if cfg['plot_pulses'] == "False":
            plot_pulses = False
        elif cfg['plot_pulses'] == "True":
            plot_pulses = True
        else:
            raise(Exception("Couldn't resolve setting of 'plot_pulses'"))
    else:
        plot_pulses = False
    opt = C1(
        dir_path=cfg['dir_path'],
        fid_func=fid_func,
        fid_subspace=cfg['fid_subspace'],
        gateset_opt_map=gateset_opt_map,
        opt_gates=opt_gates,
        callback_fids=callback_fids,
        algorithm=algorithm,
        plot_dynamics=plot_dynamics,
        plot_pulses=plot_pulses,
        options=options
    )
    return opt


def create_c1_opt_hk(
    optimizer_config,
    lindblad,
    RB_number,
    RB_length,
    shots,
    noise
):
    with open(optimizer_config, "r") as cfg_file:
        try:
            cfg = json.loads(cfg_file.read())
        except json.decoder.JSONDecodeError:
            raise Exception(f"Config {optimizer_config} is invalid.")

    if lindblad:
        def unit_X90p(U_dict):
            return fidelities.lindbladian_unitary_infid(U_dict, 'X90p', proj=True)
        def avfid_X90p(U_dict):
            return fidelities.lindbladian_average_infid(U_dict, 'X90p', proj=True)
        def epc_ana(U_dict):
            return fidelities.lindbladian_epc_analytical(U_dict, proj=True)
    else:
        def unit_X90p(U_dict):
            return fidelities.unitary_infid(U_dict, 'X90p', proj=True)
        # def unit_Y90p(U_dict):
        #     return fidelities.unitary_infid(U_dict, 'Y90p', proj=True)
        # def unit_X90m(U_dict):
        #     return fidelities.unitary_infid(U_dict, 'X90m', proj=True)
        # def unit_Y90m(U_dict):
        #     return fidelities.unitary_infid(U_dict, 'Y90m', proj=True)
        def avfid_X90p(U_dict):
            return fidelities.average_infid(U_dict, 'X90p', proj=True)
        def epc_ana(U_dict):
            return fidelities.epc_analytical(U_dict, proj=True)
    seqs = qt_utils.single_length_RB(RB_number=RB_number, RB_length=RB_length)
    def orbit_no_noise(U_dict):
        return fidelities.orbit_infid(U_dict, lindbladian=lindblad,
            seqs=seqs)
    def orbit_seq_noise(U_dict):
        return fidelities.orbit_infid(U_dict, lindbladian=lindblad,
            RB_number=RB_number, RB_length=RB_length)
    def orbit_shot_noise(U_dict):
        return fidelities.orbit_infid(U_dict, lindbladian=lindblad,
            seqs=seqs, shots=shots, noise=noise)
    def orbit_seq_shot_noise(U_dict):
        return fidelities.orbit_infid(U_dict,lindbladian=lindblad,
            shots=shots, noise=noise,
            RB_number=RB_number, RB_length=RB_length)
    def epc_RB(U_dict):
        return fidelities.RB(U_dict, logspace=True, lindbladian=lindblad)[0]
    def epc_leakage_RB(U_dict):
        return fidelities.leakage_RB(U_dict,
            logspace=True, lindbladian=lindblad)[0]
    seqs100 = qt_utils.single_length_RB(RB_number=100, RB_length=RB_length)
    def maw_orbit(U_dict):
        sampled_seqs = random.sample(seqs100, k=RB_number)
        return fidelities.orbit_infid(U_dict, lindbladian=lindblad,
            seqs=sampled_seqs, shots=shots, noise=noise)

    fids = {
        'unitary_infid': unit_X90p,
        # 'unitary_infid_Y90p': unit_Y90p,
        # 'unitary_infid_X90m': unit_X90m,
        # 'unitary_infid_Y90m': unit_Y90m,
        'average_infid': avfid_X90p,
        'orbit_no_noise': orbit_no_noise,
        'orbit_seq_noise': orbit_seq_noise,
        'orbit_shot_noise': orbit_shot_noise,
        'orbit_seq_shot_noise': orbit_seq_shot_noise,
        'maw_orbit': maw_orbit,
        'epc_RB': epc_RB,
        'epc_leakage_RB': epc_leakage_RB,
        'epc_ana': epc_ana
    }
    fid = cfg['fid_func']
    cb_fids = cfg['callback_fids']
    fid_func = fids[fid]
    callback_fids = []
    for cb_fid in cb_fids:
        callback_fids.append(fids[cb_fid])
    gateset_opt_map = [
        [tuple(par) for par in set]
        for set in cfg['gateset_opt_map']
    ]
    algorithm = algorithms[cfg['algorithm']]
    options = {}
    if 'options' in cfg:
        options = cfg['options']
    opt = C1(
        dir_path=cfg['dir_path'],
        fid_func=fid_func,
        gateset_opt_map=gateset_opt_map,
        callback_fids=callback_fids,
        algorithm=algorithm,
        options=options
    )
    return opt


def create_c2_opt(optimizer_config, eval_func_path):
    with open(optimizer_config, "r") as cfg_file:
        try:
            cfg = json.loads(cfg_file.read())
        except json.decoder.JSONDecodeError:
            raise Exception(f"Config {optimizer_config} is invalid.")

    exp_eval_namespace = run_path(eval_func_path)

    try:
        exp_type = cfg['exp_type']
    except KeyError:
        raise Exception(
            "C3:ERROR:No experiment type found in "
            f"{optimizer_config}"
        )
    try:
        eval_func = exp_eval_namespace[exp_type]
    except KeyError:
        raise Exception(
            f"C3:ERROR:Unkown experiment type: {cfg['exp_type']}"
        )

    gateset_opt_map = [
        [tuple(par) for par in set]
        for set in cfg['gateset_opt_map']
    ]
    state_labels = None
    if 'state_labels' in cfg:
        state_labels = cfg["state_labels"]
    logdir = cfg['dir_path'] + 'RB_c2_' + time.strftime(
        "%Y_%m_%d_T_%H_%M_%S/", time.localtime()
    )
    # if not os.path.isdir(logdir):
    #     os.makedirs(logdir)
    if 'exp_right' in exp_eval_namespace:
        exp_right = exp_eval_namespace['exp_right']
        def eval(p):
            return eval_func(
                p, exp_right, gateset_opt_map, state_labels, logdir
            )
    else:
        eval = eval_func
    algorithm = algorithms[cfg['algorithm']]
    options = {}
    if 'options' in cfg:
        options = cfg['options']
    opt = C2(
        dir_path=cfg['dir_path'],
        eval_func=eval,
        gateset_opt_map=gateset_opt_map,
        algorithm=algorithm,
        options=options
    )
    return opt, exp_right


def create_c3_opt(optimizer_config):
    with open(optimizer_config, "r") as cfg_file:
        cfg = json.loads(cfg_file.read())

    state_labels = {"all": None}
    if "state_labels" in cfg:
        for target, labels in cfg["state_labels"].items():
            state_labels[target] = [tuple(l) for l in labels]


    try:
        estimator = cfg['estimator']
    except KeyError:
        print(
            "C3:WARNING: Non estimator given."
            " Using default estimator RMS distance."
        )
        estimator = 'rms_dist'
    try:
        fom = estimators[estimator]
    except KeyError:
        print(
            f"C3:WARNING: No estimator named \'{estimator}\' found."
            " Using default estimator RMS distance."
        )
        fom = estimators['rms_dist']

    try:
        cb_foms = cfg['callback_est']
    except KeyError:
        print("C3:WARNING: Non callback estimators given.")
        cb_foms = []

    callback_foms = []
    for cb_fom in cb_foms:
        try:
            callback_foms.append(estimators[cb_fom])
        except KeyError:
            print(
                f"C3:WARNING: No estimator named \'{estimator}\' found."
                " Skipping this callback estimator."
            )

    callback_figs = []
    for key in cfg['callback_figs']:
        callback_figs.append(plots[key])
    exp_opt_map = [tuple(a) for a in cfg['exp_opt_map']]
    try:
        algorithm = algorithms[cfg['algorithm']]
    except KeyError:
        raise KeyError("C3:ERROR:Unkown Algorithm.")

    try:
        sampling_func = sampling[cfg['sampling']]
    except KeyError:
        raise KeyError("C3:ERROR:Unkown sampling method.")

    options = {}
    if 'options' in cfg:
        options = cfg['options']

    batch_sizes = cfg['batch_size']

    opt = C3(
        dir_path=cfg['dir_path'],
        fom=fom,
        sampling=sampling_func,
        batch_sizes=batch_sizes,
        opt_map=exp_opt_map,
        state_labels=state_labels,
        callback_foms=callback_foms,
        callback_figs=callback_figs,
        algorithm=algorithm,
        options=options,
    )
    return opt

def create_sensitivity_test(task_config):
    with open(task_config, "r") as cfg_file:
        cfg = json.loads(cfg_file.read())


    state_labels = {"all": None}
    if "state_labels" in cfg:
        for target, labels in cfg["state_labels"].items():
            state_labels[target] = [tuple(l) for l in labels]


    try:
        estimator = cfg['estimator']
    except KeyError:
        print(
            "C3:WARNING: Non estimator given."
            " Using default estimator RMS distance."
        )
        estimator = 'rms_dist'
    try:
        fom = estimators[estimator[0]]
    except KeyError:
        print(
            f"C3:WARNING: No estimator named \'{estimator}\' found."
            " Using default estimator RMS distance."
        )
        fom = estimators['neg_loglkh_gauss_norm']

    try:
        cb_foms = cfg['callback_est']
    except KeyError:
        print("C3:WARNING: Non callback estimators given.")
        cb_foms = []

    callback_foms = []
    for cb_fom in cb_foms:
        try:
            callback_foms.append(estimators[cb_fom])
        except KeyError:
            print(
                f"C3:WARNING: No estimator named \'{estimator}\' found."
                " Skipping this callback estimator."
            )

    callback_figs = []
    for key in cfg['callback_figs']:
        callback_figs.append(plots[key])
    exp_opt_map = [tuple(a) for a in cfg['exp_opt_map']]
    try:
        algorithm = algorithms[cfg['algorithm']]
    except KeyError:
        raise KeyError("C3:ERROR:Unkown Algorithm.")

    try:
        sampling_func = sampling[cfg['sampling']]
    except KeyError:
        raise KeyError("C3:ERROR:Unkown sampling method.")




    sweep_map = []
    for a in cfg['sweep_map']:
        tmp = []
        tmp.append(eval(a[2][0]))
        tmp.append(eval(a[2][1]))
        sweep_map.append((a[0],a[1],tuple(tmp)))

    if 'accuracy_goal' in cfg:
        accuracy_goal = cfg['accuracy_goal']

    if 'probe_list' in cfg:
        probe_list = []
        for x in cfg['probe_list']:
            probe_list.append(eval(x))



    options = {}


    batch_sizes = cfg['batch_size']

    if 'options' in cfg:
        options = cfg['options']
    set = SET(
        dir_path=cfg['dir_path'],
        estimator_list = cfg['estimator'],
        fom=fom,
        sampling=sampling_func,
        batch_sizes=batch_sizes,
        opt_map=exp_opt_map,
        state_labels=state_labels,
        sweep_map=sweep_map,
        probe_list=probe_list,
        accuracy_goal=accuracy_goal,
        callback_foms=callback_foms,
        callback_figs=callback_figs,
        options=options
    )
    return set
