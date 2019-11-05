import json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rc
rc('font', **{'family': 'sans-serif', 'sans-serif': ['Helvetica']})
# for Palatino and other serif fonts use:
# rc('font',**{'family':'serif','serif':['Palatino']})
rc('text', usetex=True)


def plot_logs(logfilename):
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
                    else:
                        p_val = param[1]
                    if not(p_name in parameters.keys()):
                        parameters[p_name] = []
                    parameters[p_name].append(p_val)
                    units[p_name] = unit
    n_params = len(parameters.keys())
    if n_params > 0:
        nrows = np.ceil(np.sqrt(n_params + 1))
        ncols = np.ceil(n_params / nrows)
        plt.figure(figsize=(6 * nrows, 5 * ncols))
        ii = 1
        for key in parameters.keys():
            plt.subplot(nrows, ncols, ii)
            plt.plot(parameters[key])
            plt.grid()
            plt.title(key.replace('_', '\_'))
            plt.ylabel(units[key])
            ii += 1
        plt.subplot(nrows, ncols, ii)
        plt.title("Goal")
        plt.grid()
        plt.semilogy(goal_function)
