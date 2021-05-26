"""Parsers to read in config files and construct the corresponding objects."""

import hjson
import time
import random

from c3.c3objs import hjson_decode
from c3.libraries.algorithms import algorithms
from c3.libraries.estimators import estimators
from c3.libraries.fidelities import fidelities
from c3.libraries.sampling import sampling
from c3.optimizers.c1 import C1
from c3.optimizers.c2 import C2
from c3.optimizers.c3 import C3
from c3.optimizers.sensitivity import SET

# flake8: noqa: C901


def create_c1_opt(optimizer_config, exp):
    """
    Create an object for C1 optimal control.

    Parameters
    ----------
    optimizer_config : str
        File path to a hjson file containing the C1 configuration.
    lindblad : boolean
        Include lindbladian dynamics.
    Returns
    -------
    C1
        Open loop optimizer object

    """
    parameter_map = exp.pmap
    lindblad = parameter_map.model.lindbladian

    with open(optimizer_config, "r") as cfg_file:
        cfg = hjson.loads(cfg_file.read())

    if lindblad:
        fid = "lindbladian_" + cfg["fid_func"]
    else:
        fid = cfg["fid_func"]

    callback_fids = []
    if "callback_fids" in cfg:
        if lindblad:
            cb_fids = ["lindbladian_" + f for f in cfg["callback_fids"]]
        else:
            cb_fids = cfg["callback_fids"]
        for cb_fid in cb_fids:
            try:
                cb_fid_func = fidelities[cb_fid]
            except KeyError:
                raise Exception(f"C3:ERROR:Unkown goal function: {cb_fid}")
            print(f"C3:STATUS:Found {cb_fid} in libraries.")
            callback_fids.append(cb_fid_func)

    try:
        fid_func = fidelities[fid]
    except KeyError:
        raise Exception(f"C3:ERROR:Unkown goal function: {fid} ")
    print(f"C3:STATUS:Found {fid} in libraries.")

    exp.set_opt_gates(cfg["opt_gates"])
    gateset_opt_map = [[tuple(par) for par in pset] for pset in cfg["gateset_opt_map"]]
    parameter_map.set_opt_map(gateset_opt_map)

    algorithm = algorithms[cfg["algorithm"]]
    options = {}
    if "options" in cfg:
        options = cfg["options"]
    if "plot_dynamics" in cfg:
        if cfg["plot_dynamics"] == "False":
            plot_dynamics = False
        elif cfg["plot_dynamics"] == "True":
            plot_dynamics = True
        else:
            raise (Exception("Couldn't resolve setting of 'plot_dynamics'"))
    else:
        plot_dynamics = False
    if "plot_pulses" in cfg:
        if cfg["plot_pulses"] == "False":
            plot_pulses = False
        elif cfg["plot_pulses"] == "True":
            plot_pulses = True
        else:
            raise (Exception("Couldn't resolve setting of 'plot_pulses'"))
    else:
        plot_pulses = False
    if "store_unitaries" in cfg:
        if cfg["store_unitaries"] == "False":
            store_unitaries = False
        elif cfg["store_unitaries"] == "True":
            store_unitaries = True
        else:
            raise (Exception("Couldn't resolve setting of 'plot_dynamics'"))
    else:
        store_unitaries = False
    run_name = None
    if "run_name" in cfg:
        run_name = cfg["run_name"]
    opt = C1(
        dir_path=cfg["dir_path"],
        fid_func=fid_func,
        fid_subspace=cfg["fid_subspace"],
        pmap=parameter_map,
        callback_fids=callback_fids,
        algorithm=algorithm,
        store_unitaries=store_unitaries,
        options=options,
        run_name=run_name,
    )
    return opt


def create_c2_opt(optimizer_config, eval_func_path):
    """
    Create a C2 Calibration object. Can be used to simulate the calibration process, if
    the eval_func_path contains a ''real'' experiment.

    Parameters
    ----------
    optimizer_config : str
        File path to a hjson configuration file.
    eval_func_path : str
        File path to a python script, containing the functions used perform an
        experiment.

    Returns
    -------
    C2, Experiment
        The C2 optimizer and, in the case of simulated calibration, the ''real''
        experiment object.

    """
    with open(optimizer_config, "r") as cfg_file:
        try:
            cfg = hjson.loads(cfg_file.read())
        except hjson.decoder.HjsonDecodeError as hjerr:
            raise Exception(f"Config {optimizer_config} is invalid.") from hjerr

    exp_eval_namespace = run_path(eval_func_path)

    try:
        exp_type = cfg["exp_type"]
    except KeyError:
        raise Exception("C3:ERROR:No experiment type found in " f"{optimizer_config}")
    try:
        eval_func = exp_eval_namespace[exp_type]
    except KeyError as kerr:
        raise Exception(f"C3:ERROR:Unkown experiment type: {cfg['exp_type']}") from kerr

    run_name = None
    if "run_name" in cfg:
        run_name = cfg["run_name"]

    gateset_opt_map = [[tuple(par) for par in pset] for pset in cfg["gateset_opt_map"]]
    state_labels = None
    if "state_labels" in cfg:
        state_labels = cfg["state_labels"]
    logdir = (
        cfg["dir_path"]
        + "RB_c2_"
        + time.strftime("%Y_%m_%d_T_%H_%M_%S/", time.localtime())
    )
    # if not os.path.isdir(logdir):
    #     os.makedirs(logdir)
    if "exp" in exp_eval_namespace:
        exp = exp_eval_namespace["exp"]

        def eval(p):
            return eval_func(p, exp, gateset_opt_map, state_labels, logdir)

    else:
        eval = eval_func
    algorithm = algorithms[cfg["algorithm"]]
    options = {}
    if "options" in cfg:
        options = cfg["options"]
    opt = C2(
        dir_path=cfg["dir_path"],
        run_name=run_name,
        eval_func=eval,
        gateset_opt_map=gateset_opt_map,
        algorithm=algorithm,
        exp_right=exp,
        options=options,
    )
    return opt


def create_c3_opt(optimizer_config):
    """
    The optimizer object for C3 model learning, or characterization.

    Parameters
    ----------
    optimizer_config : str
        Path to the hjson configuration file.

    """
    with open(optimizer_config, "r") as cfg_file:
        cfg = hjson.loads(cfg_file.read())

    state_labels = {"all": None}
    if "state_labels" in cfg:
        for target, labels in cfg["state_labels"].items():
            state_labels[target] = [tuple(l) for l in labels]

    if "estimator" in cfg:
        raise Exception(
            f"C3:ERROR: Setting estimators is currently not supported."
            "Only the standard logarithmic likelihood can be used at the moment."
            "Please remove this setting."
        )

    try:
        cb_foms = cfg["callback_est"]
    except KeyError:
        print("C3:WARNING: Unknown callback estimators given.")
        cb_foms = []

    callback_foms = []
    for cb_fom in cb_foms:
        try:
            callback_foms.append(estimators[cb_fom])
        except KeyError:
            print(
                f"C3:WARNING: No estimator named '{cb_fom}' found."
                " Skipping this callback estimator."
            )

    exp_opt_map = [tuple(a) for a in cfg["exp_opt_map"]]

    try:
        algorithm = algorithms[cfg["algorithm"]]
    except KeyError:
        raise KeyError("C3:ERROR:Unkown Algorithm.")

    try:
        sampling_func = sampling[cfg["sampling"]]
    except KeyError:
        raise KeyError("C3:ERROR:Unkown sampling method.")

    options = {}
    if "options" in cfg:
        options = cfg["options"]

    batch_sizes = cfg["batch_size"]

    if "seqs_per_point" in cfg:
        seqs_per_point = cfg["seqs_per_point"]
    else:
        seqs_per_point = None

    run_name = None
    if "run_name" in cfg:
        run_name = cfg["run_name"]
    opt = C3(
        dir_path=cfg["dir_path"],
        sampling=sampling_func,
        batch_sizes=batch_sizes,
        seqs_per_point=seqs_per_point,
        opt_map=exp_opt_map,
        state_labels=state_labels,
        callback_foms=callback_foms,
        callback_figs=callback_figs,
        algorithm=algorithm,
        options=options,
        run_name=run_name,
    )
    return opt


def create_sensitivity(task_config):
    """
    Create the object to perform a sensitivity analysis.

    Parameters
    ----------
    task_config : str
        File path to the hjson configuration file.

    Returns
    -------
        Sensitivity object.

    """
    with open(task_config, "r") as cfg_file:
        cfg = hjson.loads(cfg_file.read())

    sweep_map = [tuple(a) for a in cfg["sweep_map"]]

    state_labels = {"all": None}
    if "state_labels" in cfg:
        for target, labels in cfg["state_labels"].items():
            state_labels[target] = [tuple(l) for l in labels]

    try:
        estimator = cfg["estimator"]
    except KeyError:
        print(
            "C3:WARNING: No estimator given." " Using default estimator RMS distance."
        )
        estimator = "rms_dist"
    try:
        fom = estimators[estimator]
    except KeyError:
        print(
            f"C3:WARNING: No estimator named '{estimator}' found."
            " Using default estimator RMS distance."
        )
        fom = estimators["rms_dist"]

    try:
        algorithm = algorithms[cfg["algorithm"]]
    except KeyError:
        raise KeyError("C3:ERROR:Unkown Algorithm.")

    try:
        sampling_func = sampling[cfg["sampling"]]
    except KeyError:
        raise KeyError("C3:ERROR:Unkown sampling method.")

    try:
        est_list = cfg["estimator_list"]
    except KeyError:
        print("C3:WARNING: No estimators given. Using RMS")
        est_list = ["rms_dist"]

    estimator_list = []
    for est in est_list:
        try:
            estimator_list.append(estimators[est])
        except KeyError:
            print(
                f"C3:WARNING: No estimator named '{est}' found."
                " Skipping this estimator."
            )

    batch_sizes = cfg["batch_size"]

    options = {}
    if "options" in cfg:
        options = cfg["options"]

    sweep_bounds = []
    for a in cfg["sweep_bounds"]:
        sweep_bounds.append([eval(a[0]), eval(a[1])])

    if "same_dyn" in cfg:
        same_dyn = bool(cfg["same_dyn"])
    else:
        same_dyn = False

    set = SET(
        dir_path=cfg["dir_path"],
        fom=fom,
        estimator_list=estimator_list,
        sampling=sampling_func,
        batch_sizes=batch_sizes,
        state_labels=state_labels,
        sweep_map=sweep_map,
        sweep_bounds=sweep_bounds,
        algorithm=algorithm,
        same_dyn=same_dyn,
        options=options,
    )
    return set
