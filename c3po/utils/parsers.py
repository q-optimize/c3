import json
import c3po.libraries.estimators as estimators
import c3po.utils.display as display
import c3po.libraries.algorithms as algorithms
from runpy import run_path
from c3po.optimizers.c3 import C3


def create_experiment(exp_setup, datafile):
    exp_namespace = run_path(exp_setup)
    exp = exp_namespace['create_experiment'](datafile)
    return exp


def create_optimizer(optimizer_config):
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
