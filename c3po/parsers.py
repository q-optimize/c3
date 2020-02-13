
import c3po.estimators as estimators
import c3po.display as display
from c3po.c3 import C3
from c3po.algorightms import cmaes, lbfgs


def create_experiment(exp_setup, datafile):
    import exp_setup.create_experiment
    exp = exp_setup.create_experiment(datafile)
    return exp

def create_optimizer(optimizer_config):
    estimator = 'rms'
    exp_opt_map = [
        # ('Model', 'confusion_row'),
        ('Model', 'meas_offset'),
        ('Model', 'meas_scale'),
        ('Model', 'init_temp'),
        ('Q1', 'freq'),
        ('Q1', 'anhar'),
        ('Q1', 't1'),
        ('Q1', 't2star'),
        ('v_to_hz', 'V_to_Hz'),
        # ('resp', 'rise_time')
    ]
    estims = {
        'median': estimators.median_dist,
        'rms': estimators.rms_dist,
        'stds': estimators.stds_dist,
        'gauss': estimators.neg_loglkh_gauss,
        'binom': estimators.neg_loglkh_binom,
        'rms_stds': estimators.rms_stds_dist,
        'std_diffs': estimators.std_of_diffs,
    }
    fom = estims.pop(estimator)
    callback_foms = estims.values()
    opt = C3(
        dir_path="/localdisk/c3logs/",
        fom=fom,
        sampling='even',
        batch_size=10,
        opt_map=exp_opt_map,
        callback_foms=callback_foms,
        callback_figs=[display.exp_vs_sim, display.exp_vs_sim_2d_hist],
        algorithm_no_grad=cmaes,
        algorithm_with_grad=lbfgs,
    )
    return opt
