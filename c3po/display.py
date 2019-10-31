import json
import numpy as np
import matplotlib.pyplot as plt


def plot_logs(logfilename):
    with open(logfilename, "r") as filename:
        log = filename.readlines()
    goal_function = []
    parameters = {}
    for line in log:
        if line[0] == "{":
            point = json.loads(line)
            goal_function.append(point['goal'])
            for param in point['params']:
                p_name = param[0][0] + ' ' + param[0][1]
                p_val = param[1]
                if not(p_name in parameters.keys()):
                    parameters[p_name] = []
                parameters[p_name].append(p_val)
    plt.figure()
    n_params = len(parameters.keys())
    nrows = int(np.sqrt(n_params+1))
    ncols = n_params / nrows
    ii = 1
    for key in parameters.keys():
        plt.subplot(nrows, ncols, ii)
        plt.plot(parameters[key])
        plt.title(key)
        ii += 1
    plt.subplot(nrows, ncols, ii)
    plt.semilogy(goal_function)
