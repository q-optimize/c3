import json
import c3po.libraries.estimators as estimators
import c3po.utils.display as display
import c3po.libraries.algorithms as algorithms
import c3po.libraries.fidelities as fidelities
from runpy import run_path
from c3po.optimizers.c1 import C1
from c3po.optimizers.c2 import C2
from c3po.optimizers.c3 import C3


def create_experiment(exp_setup, datafile=''):
    exp_namespace = run_path(exp_setup)
    if datafile:
        exp = exp_namespace['create_experiment'](datafile)
    else:
        exp = exp_namespace['create_experiment']()
    return exp


def create_c1_opt(optimizer_config):
    with open(optimizer_config, "r") as cfg_file:
        cfg = json.loads(cfg_file.read())

    def unit_X90p(U_dict):
        return fidelities.unitary_infid(U_dict, 'X90p', proj=True)

    def avfid_X90p(U_dict):
        return fidelities.average_infid(U_dict, 'X90p', proj=True)

    fids = {
        'unitary_infid': unit_X90p,
        'average_infid': avfid_X90p
    }
    fid = cfg['fid_func']
    fid_func = fids.pop(fid)
    callback_fids = fids.values()
    gateset_opt_map = [[tuple(a)] for a in cfg['gateset_opt_map']]
    grad_algs = {'lbfgs': algorithms.lbfgs}
    no_grad_algs = {'cmaes': algorithms.cmaes}
    if cfg['algorithm'] in grad_algs.keys():
        algorithm_with_grad = grad_algs[cfg['algorithm']]
        algorithm_no_grad = None
    elif cfg['algorithm'] in no_grad_algs.keys():
        algorithm_no_grad = no_grad_algs[cfg['algorithm']]
        algorithm_with_grad = None
    opt = C1(
        dir_path=cfg['dir_path'],
        fid_func=fid_func,
        gateset_opt_map=gateset_opt_map,
        callback_fids=callback_fids,
        algorithm_no_grad=algorithm_no_grad,
        algorithm_with_grad=algorithm_with_grad,
    )
    return opt


def create_c2_opt(optimizer_config):
    with open(optimizer_config, "r") as cfg_file:
        cfg = json.loads(cfg_file.read)
    exp_eval_namespace = run_path(cfg['eval_func'])
    eval_func = exp_eval_namespace['eval_func']
    gateset_opt_map = [tuple(a) for a in cfg['gateset_opt_map']]
    no_grad_algs = {'cmaes': algorithms.cmaes}
    algorithm_no_grad = no_grad_algs[cfg['algorithm']]
    opt = C2(
        dir_path=cfg['dir_path'],
        eval_func=eval_func,
        gateset_opt_map=gateset_opt_map,
        algorithm_no_grad=algorithm_no_grad,
    )
    return opt


def create_synthetic_c2_opt(optimizer_config, exp):
    with open(optimizer_config, "r") as cfg_file:
        cfg = json.loads(cfg_file.read)

    def unit_X90p(U_dict):
        return fidelities.unitary_infid(U_dict, 'X90p', proj=True)

    def avfid_X90p(U_dict):
        return fidelities.average_infid(U_dict, 'X90p', proj=True)

    fids = {
        'unitary_infid': unit_X90p,
        'average_infid': avfid_X90p
    }
    eval_func = fids.pop(cfg['fid_func'])
    gateset_opt_map = [tuple(a) for a in cfg['gateset_opt_map']]
    no_grad_algs = {'cmaes': algorithms.cmaes}
    algorithm_no_grad = no_grad_algs[cfg['algorithm']]
    opt = C2(
        dir_path=cfg['dir_path'],
        eval_func=None,
        gateset_opt_map=gateset_opt_map,
        algorithm_no_grad=algorithm_no_grad,
    )
    return opt, eval_func


def create_c3_opt(optimizer_config):
    with open(optimizer_config, "r") as cfg_file:
        cfg = json.loads(cfg_file.read())
    estimator = cfg['estimator']
    estims = {
        'median': estimators.median_dist,
        'rms': estimators.rms_dist,
        'stds': estimators.exp_stds_dist,
        'gauss': estimators.neg_loglkh_gauss,
        'binom': estimators.neg_loglkh_binom,
        'rms_stds': estimators.rms_exp_stds_dist,
        'std_diffs': estimators.std_of_diffs,
    }
    fom = estims.pop(estimator)
    callback_foms = estims.values()
    figs = {
        'exp_vs_sim': display.exp_vs_sim,
        'exp_vs_sim_2d_hist': display.exp_vs_sim_2d_hist
    }
    callback_figs = []
    for key in cfg['callback_figs']:
        callback_figs.append(figs[key])
    exp_opt_map = [tuple(a) for a in cfg['exp_opt_map']]
    grad_algs = {'lbfgs': algorithms.lbfgs}
    no_grad_algs = {'cmaes': algorithms.cmaes}
    if cfg['algorithm'] in grad_algs.keys():
        algorithm_with_grad = grad_algs[cfg['algorithm']]
        algorithm_no_grad = None
    elif cfg['algorithm'] in no_grad_algs.keys():
        algorithm_no_grad = no_grad_algs[cfg['algorithm']]
        algorithm_with_grad = None
    opt = C3(
        dir_path=cfg['dir_path'],
        fom=fom,
        sampling=cfg['sampling'],
        batch_size=int(cfg['batch_size']),
        opt_map=exp_opt_map,
        callback_foms=callback_foms,
        callback_figs=callback_figs,
        algorithm_no_grad=algorithm_no_grad,
        algorithm_with_grad=algorithm_with_grad,
    )
    return opt
