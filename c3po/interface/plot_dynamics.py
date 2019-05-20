import matplotlib.pyplot as plt


def plot_dynamics(u_list, ts, states):
    pop = []
    for si in states:
        for ti in range(len(ts)):
            pop[ti] = abs(u_list[ti][si] ** 2)
        plt.plot(ts, pop)
