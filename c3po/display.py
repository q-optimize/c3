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
            if 'goal' in point.keys():
                goal_function.append(point['goal'])
                for param in point['params']:
                    p_name = ''
                    for desc in param[0][0]:
                        p_name += ' ' + desc
                    p_val = param[1]
                    if not(p_name in parameters.keys()):
                        parameters[p_name] = []
                    parameters[p_name].append(p_val)
    n_params = len(parameters.keys())
    if n_params > 0:
        nrows = np.ceil(np.sqrt(n_params + 1))
        ncols = np.ceil(n_params / nrows)
        plt.figure(figsize=(12, 8))
        ii = 1
        for key in parameters.keys():
            plt.subplot(nrows, ncols, ii)
            plt.plot(parameters[key])
            plt.grid()
            plt.title(key)
            ii += 1
        plt.subplot(nrows, ncols, ii)
        plt.title("Goal")
        plt.grid()
        plt.plot(goal_function)
