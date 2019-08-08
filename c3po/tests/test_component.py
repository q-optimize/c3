from c3po.signals.component import Component as Comp
from c3po.signals.envelopes import *

import numpy as np
from scipy.special import erf
import matplotlib.pyplot as plt



test = Comp()
print("test uuid: " + str(test.get_uuid()))

test2 = Comp()
print("test2 uuid: " + str(test2.get_uuid()))




def gaussian(t, params):
    """
    Normalized gaussian
    """
    sigma = params['sigma']
    T_final = params['T_final']
    gauss = np.exp(-(t - T_final / 2) ** 2 / (2 * sigma ** 2)) - \
        np.exp(-T_final ** 2 / (8 * sigma ** 2))
    norm = np.sqrt(2 * np.pi * sigma ** 2) \
        * erf(T_final / (np.sqrt(8) * sigma)) \
        - T_final * np.exp(-T_final ** 2 / (8 * sigma ** 2))
    # the erf factor takes care of cutoffs at the tails of the gaussian
    return gauss / norm


def my_flattop(t, params):
    t_up = params['T_up']
    t_down = params['T_down']
    return flattop(t, t_up, t_down)


gauss_params = {
    'T_final' : 10e-9,
    'sigma' : 2.0e-9
}


flattop_params = {
    'T_up' : 2.5e-9,
    'T_down' : 7.5e-9
}


t = np.linspace(-10e-9, 10e-9, 100)

p1 = Comp(desc = "pulse1", shape = gaussian, params = gauss_params)
print("p1 uuid: " + str(p1.get_uuid()))





p2 = Comp(desc = "pulse2", shape = my_flattop, params = flattop_params)
print("p2 uuid: " + str(p2.get_uuid()))

print("set new uuid in p2")
p2.set_uuid(p1.get_uuid())
print("p2 uuid: " + str(p2.get_uuid()))


# plt.plot(t, p1.get_shape_values(t))
# plt.show()


# plt.plot(t, p2.get_shape_values(t))
# plt.show()

