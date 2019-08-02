from c3po.signals.component import Component as Comp
from c3po.signals.envelopes import *

import numpy as np
from scipy.special import erf
import matplotlib.pyplot as plt



test = Comp()
print("test id: " + str(test.get_id()))

test2 = Comp()
print("test2 id: " + str(test2.get_id()))




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
print("p1 id: " + str(p1.get_id()))


p2 = Comp(desc = "pulse2", shape = flattop, params = flattop_params)
print("p2 id: " + str(p2.get_id()))

plt.plot(t, p1.get_shape_values(t))
plt.show()


plt.plot(t, p2.get_shape_values(t))
plt.show()

