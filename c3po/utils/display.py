""" Abandon all hope, ye who enter here. """

import os
import json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rc
from matplotlib import colors as clrs
from matplotlib.ticker import MaxNLocator
from matplotlib.widgets import Slider
from c3po.utils.utils import eng_num
from IPython.display import clear_output
import warnings
import glob
warnings.filterwarnings("ignore", category=RuntimeWarning)
rc('font', **{'family': 'sans-serif', 'sans-serif': ['Helvetica']})
# for Palatino and other serif fonts use:
# rc('font',**{'family':'serif','serif':['Palatino']})
rc('text', usetex=True)

plots = dict()


def plots_reg_deco(func):
    """
    Decorator for making registry of functions
    """
    plots[str(func.__name__)] = func
    return func


def layout(n):
    """
    For a given number of plots, return a pleasing 2D arrangement.

    Parameters
    ----------
    n : int
        Number of plots

    Returns
    -------
    tuple of int
        Number of rows and columns
    """
    if not (n % 2):
        return n//2, 2
    elif n == 9:
        return 3, 3
    else:
        return n, 1


nice_parameter_name = {
    "amp": "Amplitude",
    "freq": "Frequency $\\omega_q$",
    "anhar": "Anharmonicity $\\delta$",
    "v_to_hz": "$\\Phi$",
    "V_to_Hz": "Line response",
    "freq_offset": "Detuning $\\delta\\omega_d$",
    "delta": "DRAG parameter $\\eta$",
    "t_final": "$t_{final}$",
    "t1": "$T_{1}$",
    "t2star": "$T_{2}^*$",
    "xy_angle": "$\\phi_{xy}$",
    "Q1": "Qubit 1",
    "Q2": "Qubit 2",
    "init_ground" : "",
    "init_temp": "temperature $T$",
    "conf_matrix": "$M$",
    "confusion_row_Q1": "Confusion",
    "confusion_row_Q2": "Confusion",
    "meas_rescale": "Readout",
    "meas_offset": "offset",
    "meas_scale": "scale",
    "Q1-Q2": "coupling",
    "strength": "coupling $g$",
    "X90p:Id": "$X_{+\\frac{\\pi}{2}}\\otimes\\mathcal{I}$",
    "Id:X90p": "$\\mathcal{I}\\otimes X_{+\\frac{\\pi}{2}}$",
    "t_up": "$T_{up}$",
    "t_down": "$T_{down}$"
}


def unit_conversion(desc, param):
    # TODO Get right units from the log
    use_prefix = True
    for key, item in nice_parameter_name.items():
        # Yes, this is that stupid
        if item==desc:
            desc = key
    for ii in range(2):
        if desc == "freq_offset":
            p_val = param / 2 / np.pi
            unit = 'Hz'
        elif desc == "xy_angle":
            p_val = param / np.pi
            unit = '[$\\pi$]'
            use_prefix = False
        elif desc == 'freq':
            p_val = param / 2 / np.pi
            unit = 'Hz'
        elif desc == 'strength':
            p_val = param / 2 / np.pi
            unit = 'Hz'
        elif desc == 'anhar':
            p_val = param / 2 / np.pi
            unit = 'Hz'
        elif desc == 'delta':
            p_val = param
            unit = ''
            use_prefix = False
        elif desc == 't1' or desc == 't2star':
            p_val = param
            unit = 's'
        elif desc == 'V_to_Hz':
            p_val = param
            unit = 'Hz/V'
        elif desc == "Amplitude":
            p_val = param
            unit = 'V'
        elif desc == 'rise_time':
            p_val = param
            unit = 's'
        elif desc == 'init_temp':
            p_val = param
            unit = 'K'
        elif desc == "amp":
            p_val = param
            unit = 'V'
        else:
            p_val = param
            use_prefix = False
            unit = ""
    if use_prefix:
        value, prefix = eng_num(p_val)
        return value, " ["+prefix+unit+"]"
    else:
        return p_val, unit


@plots_reg_deco
def exp_vs_sim(exps, sims, stds):
    fig = plt.figure()
    exps = np.reshape(exps, exps.shape[0])
    sims = np.reshape(sims, sims.shape[0])
    plt.scatter(exps, sims)
    plt.title('Infidelity correlation')
    plt.xlabel('Experiment')
    plt.ylabel('Simulation')
    return fig


@plots_reg_deco
def exp_vs_sim_2d_hist(exps, sims, stds):
    # example in
    # docs.scipy.org/doc/numpy/reference/generated/numpy.histogram2d.html
    n_bins = 40
    fig = plt.figure()
    exps = np.reshape(exps, exps.shape[0])
    sims = np.reshape(sims, sims.shape[0])
    n_exps, _ = np.histogram(exps, bins=n_bins)
    H, xedges, yedges = np.histogram2d(exps, sims, bins=n_bins)
    H = np.zeros([n_bins, n_bins]) + (H.T / n_exps)
    plt.imshow(
        H,
        origin='lower',
        # interpolation='bilinear',
        # extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]]
        extent=[0, 1, 0, 1],
        aspect="equal"
    )
    plt.title('Infidelity correlation')
    plt.xlabel('Experiment')
    plt.ylabel('Simulation')
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
    plt.title('Infidelity correlation')
    plt.xlabel('Experiment')
    plt.ylabel('Simulation')
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


def plot_C1(logfolder="", only_iterations=True):
    clear_output(wait=True)
    logfilename = logfolder + "open_loop.log"
    with open(logfilename, "r") as filename:
        log = filename.readlines()

    if only_iterations:
        xlabel = "Iterations"
    else:
        xlabel = "Evaluations"

    goal_function = []
    best_goal=987654321
    parameters = {}
    opt_map = json.loads(log[3])

    subplot_ids = {}
    subplot_legends = {}
    subplot_id = 1
    for line in log[4:]:
        if line[0] == "{":
            point = json.loads(line)
            if 'goal' in point.keys():
                if only_iterations and point['goal']<best_goal:
                    goal_function.append(point['goal'])
                    best_goal = point['goal']
                    for iparam in range(len(point['params'])):
                        param = point['params'][iparam]
                        unit = ''
                        p_name = ''
                        for desc in opt_map[iparam][0]:
                            try:
                                nice_name = nice_parameter_name[desc]
                            except KeyError:
                                nice_name = desc
                            p_name += '-' + nice_name
                        if not(p_name in parameters.keys()):
                            parameters[p_name] = []
                        parameters[p_name].append(param)
                        p_name_splt = p_name.split("-")
                        p_type = p_name_splt[-1]
                        par_identifier = p_name_splt[1]
                        if not p_type in subplot_ids.keys():
                            subplot_ids[p_type] = subplot_id
                            subplot_legends[p_type] = []
                            subplot_id += 1
                        if not par_identifier in subplot_legends[p_type]:
                            subplot_legends[p_type].append(par_identifier)

    scaling = {}
    units = {}
    for p_name, par in parameters.items():
        max_val = np.max(np.abs(par))
        p_val, unit = unit_conversion(p_name.split("-")[-1], max_val)
        try:
            scaling[p_name] = np.array(p_val / max_val)
        except ZeroDivisionError:
            scaling[p_name] = 1
        units[p_name] = unit

    if only_iterations:
        its = range(len(goal_function))
    else:
        its = range(1, len(goal_function) + 1)
    subplots = {}
    if len(subplot_ids) > 0:

        # Square layout
        # nrows = np.ceil(np.sqrt(len(subplot_ids)))
        # ncols = np.ceil((len(subplot_ids)) / nrows)
        # fig = plt.figure(figsize=(4 * ncols, 3 * nrows))

        # One column layout
        nrows = len(subplot_ids)
        ncols = 1
        fig, axs = plt.subplots(
            figsize=(6, 3 * nrows), nrows=nrows, ncols=ncols, sharex=True
        )
        fig.subplots_adjust(hspace=0)
        for key in parameters.keys():
            p_type = key.split("-")[-1]
            if not p_type in subplots.keys():
                subplots[p_type] = axs[subplot_ids[p_type]-1]
            ax = subplots[p_type]
            ax.plot(its, scaling[key] * parameters[key])
            ax.tick_params(
                direction="in", left=True, right=True, top=True, bottom=True
            )
            ax.set_ylabel(p_type + units[key])
            ax.xaxis.set_major_locator(MaxNLocator(integer=True))
            ax.grid(linestyle="--")

        ax.set_xlabel(xlabel)
        for p_type, legend in subplot_legends.items():
            subplots[p_type].legend(legend)
        plt.show(block=False)
        plt.savefig(logfolder + "open_loop.png")
        fig = plt.figure(figsize=(6, 4))
        plt.title("Goal")
        plt.grid()
        plt.xlabel(xlabel)
        plt.semilogy(its, goal_function)
        plt.show(block=False)
        plt.savefig(logfolder + "goal.png")


def plot_C2(cfgfolder="", logfolder=""):
    clear_output(wait=True)
    logfilename = logfolder + "calibration.log"
    if not os.path.isfile(logfilename):
        logfilename = "/tmp/c3logs/recent/calibration.log"
    with open(logfilename, "r") as filename:
        log = filename.readlines()
    goal_function = []
    batch = 0
    options = json.loads(log[6])
    batch_size = options['popsize']

    eval = 0
    for line in log[7:]:
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
    plt.show(block=False)
    plt.savefig(logfolder + "closed_loop.png")


def plot_C3(
    logfolders=["./"], change_thresh=0, only_iterations=True,
    combine_plots=False, interactive=False
):
    """
    Generates model learning plots. Default options assume the function is
    called from inside a log folder. Otherwise a file location has to be given.
    """
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

        best_goal = 987654321
        goal_function = []
        parameters = {}
        scaling = {}
        opt_map = json.loads(log[3])

        subplot_names = []
        subplot_legends = {}
        for line in log[4:]:
            if line[0] == "{":
                point = json.loads(line)
                if 'goal' in point.keys():
                    if only_iterations and point['goal']>best_goal:
                        pass
                    else:
                        best_goal = point['goal']
                        goal_function.append(point['goal'])

                        for iparam in range(len(point['params'])):
                            param = point['params'][iparam]
                            if type(param) is list:
                                param=param[0]
                            unit = ''
                            p_name = ''
                            for desc in opt_map[iparam]:
                                try:
                                    nice_name = nice_parameter_name[desc]
                                except KeyError:
                                    nice_name = desc
                                p_name += '-' + nice_name
                            if not(p_name in parameters.keys()):
                                parameters[p_name] = []
                                if use_synthetic:
                                    real_parameters[p_name] = []
                            parameters[p_name].append(param)
                            if use_synthetic:
                                real_value = real_params[
                                        synth_opt_map.index(opt_map[iparam])
                                    ]
                                if type(real_value) is list:
                                    real_value=real_value[0]
                                real_parameters[p_name].append(real_value)
                            p_name_splt = p_name.split("-")
                            p_type = p_name_splt[-1]
                            if (
                                p_type == nice_parameter_name["freq"] or
                                p_type == nice_parameter_name["anhar"]
                            ):
                                if not p_name in subplot_names:
                                    subplot_names.append(p_name)
                            else:
                                par_identifier = p_name_splt[1]
                                if not p_type in subplot_names:
                                    subplot_names.append(p_type)


        this_log = {
            "parameters": parameters,
            "goal_function": goal_function
        }
        if use_synthetic:
            this_log["real_paramters"] = real_parameters

        logs.append(this_log)

    colors = ["tab:blue", "tab:red", "tab:green"]
    for log in logs:
        parameters = log["parameters"]
        goal_function = log["goal_function"]
        units = {}
        if use_synthetic:
            real_parameters = log["real_paramters"]

        pars_to_delete = []
        for p_name, par in parameters.items():
            if change_thresh > 0 :
                rel_change = np.max(np.abs(np.diff(par))) / par[0]
                if rel_change < change_thresh:
                    pars_to_delete.append(p_name)

            max_val = np.max(np.abs(par))
            p_val, unit = unit_conversion(p_name.split("-")[-1], max_val)
            try:
                scaling[p_name] = np.array(p_val / max_val)
            except ZeroDivisionError:
                scaling[p_name] = 1
            units[p_name] = unit
        for key in pars_to_delete:
            parameters.pop(key)

        n_params = len(parameters.keys())
        its = range(1, len(goal_function) + 1)

        subplot_ids = {}
        ii = 1
        for key in sorted(subplot_names, key=lambda key: key.split("-")[-1]):
            subplot_ids[key] = ii
            ii += 1

        subplots = {}
        if len(subplot_ids) > 0:
            nrows, ncols = layout(len(subplot_ids))
            fig, axes = plt.subplots(
                figsize=(6 * ncols, 8), nrows=nrows, ncols=ncols,
                sharex='col'
            )
            ii = 1
            for p_name in parameters.keys():
                p_type = p_name.split("-")[-1]
                if (
                    p_type == nice_parameter_name["freq"] or
                    p_type == nice_parameter_name["anhar"]
                ):
                    if not p_name in subplots.keys():
                        id = subplot_ids[p_name] - 1
                        if ncols>1:
                            subplots[p_name] = axes[id // ncols][id % ncols]
                        else:
                            subplots[p_name] = axes[id % nrows]
                    ax = subplots[p_name]
                else:
                    par_identifier = p_name.split("-")[1]
                    if not p_type in subplots.keys():
                        id = subplot_ids[p_type] - 1
                        if ncols>1:
                            subplots[p_type] = axes[id // ncols][id % ncols]
                        else:
                            subplots[p_type] = axes[id % nrows]
                    ax = subplots[p_type]
                try:
                    l = ax.plot(
                        its, scaling[p_name] * parameters[p_name], ".-",
                        markersize=12, linewidth=0.5, label=par_identifier
                    )
                    if use_synthetic:
                        sim_color = l[-1].get_color()
                        real_color = clrs.hsv_to_rgb(
                            clrs.rgb_to_hsv(clrs.to_rgb(sim_color))
                            - [0, 0, 0.25]
                        )
                        ax.plot(
                            its, scaling[p_name] *  real_parameters[p_name], "--",
                            color=real_color, label = par_identifier + " real",
                            linewidth=2
                        )
                        ax.tick_params(
                            direction="in", left=True, right=True, top=True, bottom=True
                        )
                        ax.set_ylabel(p_type + units[p_name])
                        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
                        ax.grid(linestyle="--", linewidth=0.5)
                        ii += 1
                        ax.legend()
                except:
                    pass
                    #print(f"Error plotting {p_name}")
    plt.xlabel('Evaluation')
    for p_type, legend in subplot_legends.items():
        subplots[p_type].legend(legend)
    plt.tight_layout()
    plt.savefig(logfolder + "learn_model.png")
    if interactive:
        plt.show()

    plt.figure()
    plt.title("Goal")
    plt.grid()
    markers = ["x", "+", "."]
    line_names = ["simple", "intermediate", "full"]
    idx = 0
    leg_formatted = []
    for log in logs:
        goal_function = log["goal_function"]
        its = range(1, len(goal_function) + 1)
        c = colors.pop(0)
        line = plt.semilogx(
            its, goal_function, marker=markers[idx], color=c,
            label=line_names[idx]
        )
        leg_formatted.append(line)
        idx += 1
        if goal_function: #Do only if not empty
            plt.semilogx(its[-1], goal_function[-1], "x", color=c)
    leg = [fldr.replace('_', '\\_').replace("/", "") for fldr in logfolders]
    plt.legend(leg)
    plt.xlabel('Evaluation')
    plt.ylabel('model match')
    plt.tight_layout()
    plt.savefig(logfolder + "learn_model_goals.png", dpi=300)
    if interactive:
        plt.show()


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
