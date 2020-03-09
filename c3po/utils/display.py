import os
import json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rc
from matplotlib.ticker import MaxNLocator
from matplotlib.widgets import Slider
from c3po.utils.utils import eng_num
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)
rc('font', **{'family': 'sans-serif', 'sans-serif': ['Helvetica']})
# for Palatino and other serif fonts use:
# rc('font',**{'family':'serif','serif':['Palatino']})
rc('text', usetex=True)


def unit_conversion(desc, param):
    # TODO Get right units from the log
    if desc == 'freq_offset':
        p_val = param / 2 / np.pi
        unit = 'Hz'
    elif desc == 'xy_angle':
        p_val = param / np.pi
        unit = '$\\pi$'
    elif desc == 'freq':
        p_val = param / 2 / np.pi
        unit = 'Hz'
    elif desc == 'anhar':
        p_val = param / 2 / np.pi
        unit = 'Hz'
    elif desc == 't1' or desc == 't2star':
        p_val = param
        unit = 's'
    elif desc == 'V_to_Hz':
        p_val = param
        unit = 'Hz/V'
    elif desc == 'rise_time':
        p_val = param
        unit = 's'
    elif desc == 'init_temp':
        p_val = param
        unit = 'K'
    else:
        p_val = param
        unit = ""
    value, prefix = eng_num(p_val)
    return value, prefix+unit


def exp_vs_sim(exps, sims, stds):
    fig = plt.figure()
    plt.scatter(exps, sims)
    plt.title('Exp vs Sim')
    plt.xlabel('Exp fidelity')
    plt.ylabel('Sim fidelity')
    return fig


def exp_vs_sim_2d_hist(exps, sims, stds):
    # example in
    # docs.scipy.org/doc/numpy/reference/generated/numpy.histogram2d.html
    n_bins = 40
    fig = plt.figure()
    n_exps, _ = np.histogram(exps, bins=n_bins)
    H, xedges, yedges = np.histogram2d(exps, sims, bins=n_bins)
    H = (H.T / n_exps)
    plt.imshow(
        H,
        origin='lower',
        # interpolation='bilinear',
        # extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]]
        extent=[0, 1, 0, 1]
    )
    plt.title('Exp vs Sim')
    plt.xlabel('Exp fidelity')
    plt.ylabel('Sim fidelity')
    plt.colorbar()
    return fig


def get_sim_exp_std_diff(logfilename=""):
    if logfilename == "":
        # logfilename = "/tmp/c3logs/recent/confirm.log"
        logfilename = "/tmp/c3logs/recent/learn_model.log"
    with open(logfilename, "r") as filename:
        log = filename.readlines()
    sims = []
    exps = []
    stds = []
    diffs = []
    par_lines_count = 0
    for line in log[::-1]:
        if line[0] == "{":
            par_lines_count += 1
        if par_lines_count == 1 and line[:12] == '  Simulation':
            line_split = line.split()
            sims.append(np.abs(float(line_split[1])))
            exps.append(np.abs(float(line_split[3])))
            stds.append(np.abs(float(line_split[5])))
            diffs.append(np.abs(float(line_split[7])))
        elif par_lines_count == 2:
            break
    return sims, exps, stds, diffs


def plot_exp_vs_sim(logfilename=""):
    plt.figure()
    sims, exps, stds, diffs = get_sim_exp_std_diff(logfilename)
    pixel_size = (72./300) ** 2
    plt.scatter(exps, sims, s=pixel_size)
    plt.title('Exp vs Sim')
    plt.xlabel('Exp fidelity')
    plt.ylabel('Sim fidelity')
    data_path = "/".join(logfilename.split("/")[:-1])+"/"
    if data_path == "/":
        data_path = "./"
    plt.savefig(data_path+"exp_vs_sim.png", dpi=300)
    fig = exp_vs_sim_2d_hist(exps, sims, stds)
    plt.savefig(data_path+"exp_vs_sim_2d_hist.png", dpi=300)
    return fig


def plot_exp_vs_err(logfilename=""):
    plt.figure()
    sims, exps, stds, diffs = get_sim_exp_std_diff(logfilename)
    plt.scatter(exps, diffs)
    plt.title('Exp vs Diff')
    plt.xlabel('Exp fidelity')
    plt.ylabel('Sim/Exp fidelity diff')
    plt.show(block=False)


def plot_exp_vs_errstd(logfilename=""):
    plt.figure()
    sims, exps, stds, diffs = get_sim_exp_std_diff(logfilename)
    errs = []
    for indx in range(len(diffs)):
        errs.append(diffs[indx]/stds[indx])
    plt.scatter(exps, errs)
    plt.title('Exp vs Diff (in std)')
    plt.xlabel('Exp fidelity')
    plt.ylabel('Sim/Exp fidelity diff (in std)')
    plt.show(block=False)


def plot_exp_vs_std(logfilename=""):
    plt.figure()
    sims, exps, stds, diffs = get_sim_exp_std_diff(logfilename)
    errs = []
    for indx in range(len(diffs)):
        errs.append(diffs[indx]/stds[indx])
    plt.scatter(exps, stds)
    plt.title('Exp vs std')
    plt.xlabel('Exp fidelity')
    plt.ylabel('Exp fidelity std')
    plt.show(block=False)


def plot_distribution(logfilename=""):
    sims, exps, stds, diffs = get_sim_exp_std_diff(logfilename)
    plt.hist(diffs, bins=101)
    print(f"RMS: {np.sqrt(np.mean(np.square(diffs)))}")
    print(f"Median: {np.median(diffs)}")
    plt.title('distribution of difference')
    plt.show()
    return diffs


def plot_C1(logfolder=""):
    logfilename = logfolder + "open_loop.log"
    with open(logfilename, "r") as filename:
        log = filename.readlines()
    goal_function = []
    parameters = {}
    scaling = {}
    units = {}
    opt_map = json.loads(log[3])
    for line in log[4:]:
        if line[0] == "{":
            point = json.loads(line)
            if 'goal' in point.keys():
                goal_function.append(point['goal'])
                for iparam in range(len(point['params'])):
                    param = point['params'][iparam]
                    unit = ''
                    p_name = ''
                    for desc in opt_map[iparam][0]:
                        p_name += ' ' + desc
                    if p_name not in scaling:
                        p_val, unit = unit_conversion(desc, param)
                        try:
                            scaling[p_name] = p_val / param
                        except ZeroDivisionError:
                            scaling[p_name] = 1
                        units[p_name] = unit
                    if not(p_name in parameters.keys()):
                        parameters[p_name] = []
                    parameters[p_name].append(param * scaling[p_name])
    n_params = len(parameters.keys())
    its = range(1, len(goal_function) + 1)
    if n_params > 0:
        nrows = np.ceil(np.sqrt(n_params + 1))
        ncols = np.ceil((n_params + 1) / nrows)
        fig = plt.figure(figsize=(3 * ncols, 2 * nrows))
        ii = 1
        for key in parameters.keys():
            plt.subplot(nrows, ncols, ii)
            plt.plot(its, parameters[key])
            plt.grid()
            plt.title(key.replace('_', '\\_'))
            plt.ylabel(units[key])
            plt.xlabel("Evaluation")
            ii += 1
        plt.subplot(nrows, ncols, ii)
        plt.savefig(logfolder + "open_loop.png")
        plt.figure()
        plt.ylim([0.2, 0.001])
        plt.title("Goal")
        plt.grid()
        plt.xlabel("Evaluations")
        plt.plot(its, goal_function)
        plt.tight_layout()
        plt.savefig(logfolder + "goal.png")
        plt.close(fig)


def plot_C2(cfgfolder="", logfolder=""):
    logfilename = logfolder + "calibration.log"
    if not os.path.isfile(logfilename):
        logfilename = "/tmp/c3logs/recent/calibration.log"
    with open(logfilename, "r") as filename:
        log = filename.readlines()
    goal_function = []
    batch = 0
    with open(cfgfolder+"c2.cfg", "r") as cfg_file:
        cfg = json.loads(cfg_file.read())
        batch_size = cfg['options']['popsize']
    eval = 0
    for line in log[5:]:
        if line[0] == "{":
            if not eval % batch_size:
                batch = int(eval / batch_size)
                goal_function.append([])
            eval += 1
            point = json.loads(line)
            if 'goal' in point.keys():
                goal_function[batch].append(point['goal'])

    fig = plt.figure()
    plt.title("Calibration")
    means = []
    for ii in range(len(goal_function)):
        means.append(np.mean(np.array(goal_function[ii])))
        for pt in goal_function[ii]:
            plt.scatter(ii+1, pt, color='tab:blue')
    ax = plt.gca()
    # ax.set_yscale('log')
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    plt.grid()
    plt.plot(range(1, len(goal_function)+1), means, color="tab:red")
    plt.axis('tight')
    plt.ylabel('Goal function')
    plt.xlabel('Evaluations')
    plt.savefig(logfolder + "closed_loop.png")
    plt.close(fig)


def plot_C3(logfolders):
    logs = []
    for logfolder in logfolders:
        logfilename = logfolder + 'model_learn.log'
        with open(logfilename, "r") as filename:
            log = filename.readlines()

        synthetic_model = logfolder + 'real_model_params.log'

        use_synthetic = os.path.isfile(synthetic_model)

        if use_synthetic:
            with open(synthetic_model, "r") as filename:
                synth_model = filename.readlines()
            real_params = json.loads(synth_model[1])['params']
            real_parameters = {}
            synth_opt_map = json.loads(synth_model[0])

        goal_function = []
        parameters = {}
        scaling = {}
        units = {}
        opt_map = json.loads(log[3])
        for line in log[4:]:
            if line[0] == "{":
                point = json.loads(line)
                if 'goal' in point.keys():
                    goal_function.append(point['goal'])

                    for iparam in range(len(point['params'])):
                        param = point['params'][iparam]
                        unit = ''
                        p_name = ''
                        for desc in opt_map[iparam]:
                            p_name += ' ' + desc
                        if p_name not in scaling:
                            p_val, unit = unit_conversion(desc, param)
                            scaling[p_name] = p_val / param
                            units[p_name] = unit
                        if not(p_name in parameters.keys()):
                            parameters[p_name] = []
                            if use_synthetic:
                                real_parameters[p_name] = []
                        parameters[p_name].append(param * scaling[p_name])
                        if use_synthetic:
                            real_value = real_params[
                                    synth_opt_map.index(opt_map[iparam])
                                ]
                            real_parameters[p_name].append(
                                real_value * scaling[p_name]
                            )
        this_log = {
            "parameters": parameters,
            "units": units,
            "goal_function": goal_function
        }
        if use_synthetic:
            this_log["real_paramters"] = real_parameters

        logs.append(this_log)

    for log in logs:
        parameters = log["parameters"]
        units = log["units"]
        goal_function = log["goal_function"]
        if use_synthetic:
            real_parameters = log["real_paramters"]
        n_params = len(parameters.keys())
        its = range(1, len(goal_function) + 1)
        if n_params > 0:
            nrows = np.ceil(np.sqrt(n_params))
            ncols = np.ceil((n_params) / nrows)
            fig = plt.figure(figsize=(6 * ncols, 4 * nrows))
            ii = 1
            for key in parameters.keys():
                plt.subplot(nrows, ncols, ii)
                plt.plot(its, parameters[key], color='tab:blue')
                if use_synthetic:
                    plt.plot(its, real_parameters[key], "--", color='tab:red')
                plt.grid()
                plt.title(key.replace('_', '\\_'))
                plt.xlabel('Evaluation')
                plt.ylabel(units[key])
                ii += 1
                plt.tight_layout()
                plt.savefig(logfolder + "learn_model.png")
                plt.close(fig)

    fig = plt.figure()
    plt.title("Goal")
    plt.grid()
    colors = ["tab:blue", "tab:red", "tab:green"]
    line_names = ["simple", "intermediate", "full"]
    idx = 0
    leg_formatted = []
    for log in logs:
        goal_function = log["goal_function"]
        its = range(1, len(goal_function) + 1)
        c = colors.pop(0)
        line =  plt.semilogx(
            its, goal_function, color=c, label=line_names[idx]
        )
        leg_formatted.append(line)
        idx += 1
        plt.semilogx(its[-1], goal_function[-1], "x", color=c)
    leg = [fldr.replace('_', '\\_').replace("/", "") for fldr in logfolders]
  #  for entry in leg:
  #      leg_formatted.append(entry)
  #      leg_formatted.append("converged")
    plt.legend(leg_formatted)
    plt.xlabel('Evaluation')
    plt.ylabel('RMS model match')
    plt.tight_layout()
    plt.savefig(logfolder + "learn_model_goals.png", dpi=300)
    plt.close(fig)



def plot_envelope_history(logfilename):
    with open(logfilename, "r") as filename:
        log = filename.readlines()
    point = json.loads(log[-1])
    fig, ax = plt.subplots()
    plt.subplots_adjust(left=0.25, bottom=0.25)
    l1, = plt.plot(point['inphase'], lw=2)
    l2, = plt.plot(point['quadrature'], lw=2)
    # plt.legend(['inphase', 'quadrature'])
    # plt.grid()
    axit = plt.axes([0.25, 0.1, 0.65, 0.03])
    s = Slider(axit, 'Iterations', 0, len(log), valinit=len(log))

    def update(val):
        it = int(s.val)
        point = json.loads(log[it])
        l1.set_ydata(point['inphase'])
        l2.set_ydata(point['quadrature'])
        ax.autoscale()
        fig.canvas.draw_idle()
    s.on_changed(update)
    plt.show()


def plot_awg(logfolder="", num_plots=1):
    logfilename = logfolder + "awg.log"
    if not os.path.isfile(logfilename):
        logfilename = "/tmp/c3logs/recent/awg.log"
    with open(logfilename, "r") as filename:
        log = filename.readlines()
    plt.figure(figsize=(8, 2*num_plots))
    for ii in range(num_plots):
        point = json.loads(log[-ii-1])
        plt.subplot(num_plots, 1, ii+1)
        plt.plot(point['inphase'], lw=2)
        plt.plot(point['quadrature'], lw=2)
        plt.grid()
    plt.savefig(logfolder+"awg.png", dpi=300)


def plot_foms(logfolder=""):
    logfilename = logfolder + 'learn_model.log'
    if not os.path.isfile(logfilename):
        logfilename = "/tmp/c3logs/recent/learn_from.log"
    with open(logfilename, "r") as filename:
        log = filename.readlines()
    batch = -1
    foms = []
    names = [0, 0, 0, 0, 0]
    for line in log:
        split = line.split()
        if split == []:
            continue
        elif split[0] == "Starting":
            batch += 1
            foms.append([0, 0, 0, 0, 0])
            fom_id = 0
        elif split[0:2] == ['Finished', 'batch']:
            foms[batch][fom_id] = float(split[4])
            names[fom_id] = split[3].split(":")[0].replace('_', '\\_')
            fom_id += 1
    plt.semilogy(np.array(foms))
    plt.legend(names)
    plt.show(block=False)
