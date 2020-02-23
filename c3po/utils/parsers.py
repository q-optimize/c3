import json
import c3po.libraries.estimators as estimators
import c3po.utils.display as display
import c3po.libraries.algorithms as algorithms
import c3po.libraries.fidelities as fidelities
import c3po.utils.qt_utils as qt_utils
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


def create_c1_opt(
    optimizer_config,
    lindblad,
    RB_number,
    RB_length,
    shots
):
    with open(optimizer_config, "r") as cfg_file:
        cfg = json.loads(cfg_file.read())

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
            seqs=seqs, shots=shots)
    def orbit_seq_shot_noise(U_dict):
        return fidelities.orbit_infid(U_dict,lindbladian=lindblad,
            shots=shots, RB_number=RB_number, RB_length=RB_length)
    def epc_RB(U_dict):
        return fidelities.RB(U_dict, logspace=True, lindbladian=lindblad)[0]
    def epc_leakage_RB(U_dict):
        return fidelities.leakage_RB(U_dict,
            logspace=True, lindbladian=lindblad)[0]


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
    grad_algs = {'lbfgs': algorithms.lbfgs}
    no_grad_algs = {'cmaes': algorithms.cmaes}
    if cfg['algorithm'] in grad_algs.keys():
        algorithm_with_grad = grad_algs[cfg['algorithm']]
        algorithm_no_grad = None
    elif cfg['algorithm'] in no_grad_algs.keys():
        algorithm_no_grad = no_grad_algs[cfg['algorithm']]
        algorithm_with_grad = None
    options = {}
    if 'options' in cfg:
        options = cfg['options']
    opt = C1(
        dir_path=cfg['dir_path'],
        fid_func=fid_func,
        gateset_opt_map=gateset_opt_map,
        callback_fids=callback_fids,
        algorithm_no_grad=algorithm_no_grad,
        algorithm_with_grad=algorithm_with_grad,
        options=options
    )
    return opt


def create_c2_opt(optimizer_config, eval_func_path):
    with open(optimizer_config, "r") as cfg_file:
        cfg = json.loads(cfg_file.read())
    exp_eval_namespace = run_path(eval_func_path)
    eval_func = exp_eval_namespace['eval_func']
    gateset_opt_map = [
        [tuple(par) for par in set]
        for set in cfg['gateset_opt_map']
    ]
    if 'exp_right' in exp_eval_namespace:
        exp_right = exp_eval_namespace['exp_right']
        eval = lambda p: eval_func(p, exp_right, gateset_opt_map)
    else:
        eval = eval_func
    no_grad_algs = {'cmaes': algorithms.cmaes}
    algorithm_no_grad = no_grad_algs[cfg['algorithm']]
    opt = C2(
        dir_path=cfg['dir_path'],
        eval_func=eval,
        gateset_opt_map=gateset_opt_map,
        algorithm_no_grad=algorithm_no_grad,
    )
    return opt


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
