from c3po.signals.envelopes import *

from c3po.cobj.parameter import Parameter as Param
from c3po.cobj.parameter import Instance as Inst
from c3po.cobj.cfunc import CFunc as CFunc

import numpy as np
from scipy.special import erf
import matplotlib.pyplot as plt




Psigma = Param(
    string = "sigma",
    comment = "sigma of the gaussian",
    latex = "\sigma"
)


sigma = Inst(
    value = 2.0e-9,
    unit = "w/e",
    bounds = [0,5],
    param_uuid = Psigma.get_uuid()
)


Pt_final = Param(
    string = "t_final",
    comment = "t_final of gaussian",
    latex = "t_{final}",
)

t_final = Inst(
    value = 10e-9,
    unit = "sec",
    bounds = [0,5],
    param_uuid = Pt_final.get_uuid()
)



def gaussian(t, attr):
    """
    Normalized gaussian
    """
    sigma = attr["sigma"]
    t_final = attr["t_final"]

    gauss = np.exp(-(t - t_final / 2) ** 2 / (2 * sigma ** 2)) - \
        np.exp(-t_final ** 2 / (8 * sigma ** 2))

    norm = np.sqrt(2 * np.pi * sigma ** 2) \
        * erf(t_final / (np.sqrt(8) * sigma)) \
        - t_final * np.exp(-t_final ** 2 / (8 * sigma ** 2))
    # the erf factor takes care of cutoffs at the tails of the gaussian
    return gauss / norm


gauss = CFunc(
    string = "gauss",
    comment = "gaussian",
    latex = "g(t)",
    params = [Psigma, Pt_final],
    insts = [sigma, t_final],
    body = gaussian,
    body_latex = "g(t) = \frac{1}{\sqrt{2 * \pi * \sigma^2...}}\exp(...)"
)






Pt_up = Param(
    string = "t_up",
    comment = "t_up time of flattop",
    latex = "t_{up}"
)

t_up = Inst(
    value = 2.5e-9,
    unit = "sec",
    bounds = [0,5],
    param_uuid = Pt_up.get_uuid()
)


Pt_down = Param(
    string = "t_down",
    comment = "t_down time of flattop",
    latex = "t_{down}"
)


t_down = Inst(
    value = 7.5e-9,
    unit = "sec",
    bounds = [0,5],
    param_uuid = Pt_down.get_uuid()
)


def my_flattop(t, attr):
    t_up = attr['t_up']
    t_down = attr['t_down']
    return flattop(t, t_up, t_down)


flat = CFunc(
    string = "flattop",
    comment = "flattop env",
    latex = "f(t)",
    params = [Pt_up, Pt_down],
    insts = [t_up, t_down],
    body = my_flattop,
    body_latex = " ... "
)



t = np.linspace(-10e-9, 10e-9, 100)





plt.plot(t, gauss.evaluate(t))
plt.show()


plt.plot(t, flat.evaluate(t))
plt.show()

