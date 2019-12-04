import os
import json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rc
from matplotlib.ticker import MaxNLocator
from matplotlib.widgets import Slider
rc('font', **{'family': 'sans-serif', 'sans-serif': ['Helvetica']})
# for Palatino and other serif fonts use:
# rc('font',**{'family':'serif','serif':['Palatino']})
rc('text', usetex=True)


def plot_OC_logs(logfolder=""):
    logfilename = logfolder + "openloop.log"
    if not os.path.isfile(logfilename):
        logfilename = "/tmp/c3logs/recent/openloop.log"
    with open(logfilename, "r") as filename:
        log = filename.readlines()
    goal_function = []
    parameters = {}
    for line in log:
        if line[0] == "{":
            point = json.loads(line)
            if 'goal' in point.keys():
                goal_function.append(point['goal'])
                units = {}
                for param in point['params']:
                    unit = ''
                    p_name = ''
                    for desc in param[0][0]:
                        p_name += ' ' + desc
                    if desc == 'freq_offset':
                        p_val = param[1] / 1e6 / 2 / np.pi
                        unit = '[MHz]'
                    elif desc == 'xy_angle':
                        p_val = param[1] / np.pi
                        unit = '[$\\pi$]'
                    elif desc == 'freq':
                        p_val = param[1] / 1e9 / 2 / np.pi
                        unit = '[GHz]'
                    elif desc == 'anhar':
                        p_val = param[1] / 1e6 / 2 / np.pi
                        unit = '[MHz]'
                    elif desc == 'V_to_Hz':
                        p_val = param[1] / 1e6
                        unit = '[MHz/V]'
                    elif desc == 'rise_time':
                        p_val = param[1] / 1e-9
                        unit = '[ns]'
                    else:
                        p_val = param[1]
                    if not(p_name in parameters.keys()):
                        parameters[p_name] = []
                    parameters[p_name].append(p_val)
                    units[p_name] = unit
    n_params = len(parameters.keys())
    its = range(1, len(goal_function) + 1)
    if n_params > 0:
        nrows = np.ceil(np.sqrt(n_params + 1))
        ncols = np.ceil((n_params + 1) / nrows)
        plt.figure(figsize=(6 * ncols, 5 * nrows))
        ii = 1
        for key in parameters.keys():
            plt.subplot(nrows, ncols, ii)
            plt.plot(its, parameters[key])
            plt.grid()
            plt.title(key.replace('_', '\_'))
            plt.ylabel(units[key])
            plt.xlabel("Iteration")
            ii += 1
        plt.subplot(nrows, ncols, ii)
        plt.title("Goal")
        plt.grid()
        plt.xlabel("Iteration")
        plt.semilogy(its, goal_function)


def plot_calibration(logfolder=""):
    logfilename = logfolder + "calibration.log"
    if not os.path.isfile(logfilename):
        logfilename = "/tmp/c3logs/recent/calibration.log"
    with open(logfilename, "r") as filename:
        log = filename.readlines()
    goal_function = []
    batch = -1
    for line in log:
        if line[0] == "{":
            point = json.loads(line)
            if 'goal' in point.keys():
                goal_function[batch].append(point['goal'])
        elif line[0] == "B":
            goal_function.append([])
            batch += 1

    plt.figure()
    plt.title("Calibration")
    means = []
    for ii in range(len(goal_function)):
        means.append(np.mean(np.array(goal_function[ii])))
        for pt in goal_function[ii]:
            plt.scatter(ii+1, pt, color='tab:blue')
    ax = plt.gca()
    ax.set_yscale('log')
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    plt.grid()
    plt.plot(range(1, len(goal_function)+1), means, color="tab:red")
    plt.axis('tight')
    plt.ylabel('Goal function')
    plt.xlabel('Iterations')


def plot_learning(logfolder=""):
    logfilename = logfolder + 'learn_model.log'
    if not os.path.isfile(logfilename):
        logfilename = "/tmp/c3logs/recent/learn_from.log"
    with open(logfilename, "r") as filename:
        log = filename.readlines()
    goal_function = []
    parameters = {}
    for line in log:
        if line[0] == "{":
            point = json.loads(line)
            if 'goal' in point.keys():
                goal_function.append(point['goal'])
                units = {}
                for param in point['params']:
                    unit = ''
                    p_name = ''
                    for desc in param[0]:
                        p_name += ' ' + desc
                    if desc == 'freq_offset':
                        p_val = param[1] / 1e6 / 2 / np.pi
                        unit = '[MHz]'
                    elif desc == 'xy_angle':
                        p_val = param[1] / np.pi
                        unit = '[$\\pi$]'
                    elif desc == 'freq':
                        p_val = param[1] / 1e9 / 2 / np.pi
                        unit = '[GHz]'
                    elif desc == 'anhar':
                        p_val = param[1] / 1e6 / 2 / np.pi
                        unit = '[MHz]'
                    elif desc == 'V_to_Hz':
                        p_val = param[1] / 1e6
                        unit = '[MHz/V]'
                    elif desc == 'rise_time':
                        p_val = param[1] / 1e-9
                        unit = '[ns]'
                    else:
                        p_val = param[1]
                    if not(p_name in parameters.keys()):
                        parameters[p_name] = []
                    parameters[p_name].append(p_val)
                    units[p_name] = unit
    n_params = len(parameters.keys())
    its = range(1, len(goal_function) + 1)
    if n_params > 0:
        nrows = np.ceil(np.sqrt(n_params + 1))
        ncols = np.ceil((n_params + 1) / nrows)
        plt.figure(figsize=(6 * ncols, 5 * nrows))
        ii = 1
        for key in parameters.keys():
            plt.subplot(nrows, ncols, ii)
            plt.plot(its, parameters[key])
            plt.grid()
            plt.title(key.replace('_', '\_'))
            plt.ylabel(units[key])
            ii += 1
        plt.subplot(nrows, ncols, ii)
        plt.title("Goal")
        plt.grid()
        plt.semilogy(its, goal_function)


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


def plot_awg(logfolder=""):
    logfilename = logfolder + "awg.log"
    if not os.path.isfile(logfilename):
        logfilename = "/tmp/c3logs/recent/awg.log"
    with open(logfilename, "r") as filename:
        log = filename.readlines()
    point = json.loads(log[-1])
    fig, ax = plt.subplots()
    l1, = plt.plot(point['inphase'], lw=2)
    l2, = plt.plot(point['quadrature'], lw=2)
    plt.show()
