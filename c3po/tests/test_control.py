from c3po.controls.envelopes import *


from c3po.cobj.parameter import Parameter as Param
from c3po.cobj.parameter import Instance as Inst
from c3po.cobj.cfunc import CFunc as CFunc

from c3po.controls.generator import Device as Dev
from c3po.controls.generator import AWG as AWG
from c3po.controls.generator import Generator as Gen

from c3po.controls.control import Control as Ctrl


import matplotlib.pyplot as plt


Pamp = Param(
    string = "amp",
    comment = "amplitude of flattop comp",
    latex = "A"
)


Pt_up = Param(
    string = "t_up",
    comment = "t_up time of flattop",
    latex = "t_{up}"
)


Pt_down = Param(
    string = "t_down",
    comment = "t_down time of flattop",
    latex = "t_{down}"
)


Pxy_angle = Param(
    string = "xy_angle",
    comment = "xy ...",
    latex = "\Phi"
)



Pfreq_offset = Param(
    string = "freq_offset",
    comment = "freq ...",
    latex = "\omega_{xy}"
)


Pcarr_freq = Param(
    string = "carr_freq",
    comment = "carrier frequency",
    latex = "\omega_{LO}"
)


amp1 = Inst(
    value = 15e6 * 2 * np.pi,
    unit = "V", #? w/e
    param_uuid = Pamp.get_uuid()
)

t_up1 = Inst(
    value = 2.5e-9,
    unit = "sec",
    bounds = [2e-9, 98e-9],
    param_uuid = Pt_up.get_uuid()
)


t_down1 = Inst(
    value = 7.5e-9,
    unit = "sec",
    bounds = [2e-9, 98e-9],
    param_uuid = Pt_down.get_uuid()
)

xy_angle1 = Inst(
    value = 0,
    param_uuid = Pxy_angle.get_uuid()
)

freq_offset1 = Inst(
    value = 0e6 * 2 * np.pi,
    unit = " ",
    bounds = [-1e9 * 2 * np.pi, 1e9 * 2 * np.pi],
    param_uuid = Pfreq_offset.get_uuid()
)


amp2 = Inst(
    value = 3e6 * 2 * np.pi,
    unit = "V", #? w/e
    param_uuid = Pamp.get_uuid()
)

t_up2 = Inst(
    value = 25e-9,
    unit = "sec",
    bounds = [2e-9, 98e-9],
    param_uuid = Pt_up.get_uuid()
)


t_down2 = Inst(
    value = 30e-9,
    unit = "sec",
    bounds = [2e-9, 98e-9],
    param_uuid = Pt_down.get_uuid()
)

xy_angle2 = Inst(
    value = np.pi / 2.0,
    param_uuid = Pxy_angle.get_uuid()
)

freq_offset2 = Inst(
    value = 0e6 * 2 * np.pi,
    unit = " ",
    bounds = [-1e9 * 2 * np.pi, 1e9 * 2 * np.pi],
    param_uuid = Pfreq_offset.get_uuid()
)


carr_freq = Inst(
    value = 6e9 * 2 * np.pi,
    unit = "Hz", #? w/e
    param_uuid = Pcarr_freq.get_uuid()
)


def my_flattop(t, attr):
    t_up = attr['t_up']
    t_down = attr['t_down']
    return flattop(t, t_up, t_down)


flat1 = CFunc(
    string = "flat",
    comment = "flattop env",
    latex = "f_1(t)",
    params = [Pt_up, Pt_down],
    insts = [t_up1, t_down1],
    body = my_flattop,
    body_latex = " ... "
)

flat2 = CFunc(
    string = "flat",
    comment = "flattop env",
    latex = "f_2(t)",
    params = [Pt_up, Pt_down],
    insts = [t_up2, t_down2],
    body = my_flattop,
    body_latex = " ... "
)


def comp(t, attr):
    amp = attr["amp"]
    xy_angle = attr["xy_angle"]
    freq_offset = attr["freq_offset"]
    flat = attr["flat"]
    return amp * flat * np.exp(1j * (xy_angle + freq_offset * t))


comp1 = CFunc(
    string = "comp1",
    comment = "first component",
    latex = "c_1",
    params = [Pamp, Pxy_angle, Pfreq_offset],
    insts = [amp1, xy_angle1, freq_offset1],
    deps = [flat1],
    body = comp,
    body_latex = "\f(t)A\exp(...)"
)

comp2 = CFunc(
    string = "comp2",
    comment = "second component",
    latex = "c_2",
    params = [Pamp, Pxy_angle, Pfreq_offset],
    insts = [amp2, xy_angle2, freq_offset2],
    deps = [flat2],
    body = comp,
    body_latex = "\f(t)A\exp(...)"
)

####
# Below code: For checking the single signal components
####

# t = np.linspace(0, 150e-9, int(150e-9*1e9))
# plt.plot(t, flat1.evaluate_full(t)["result"])
# plt.plot(t, flat2.evaluate_full(t)["result"])
# plt.show()


# plt.plot(t, np.real(comp1.evaluate(t)))
# plt.plot(t, np.imag(comp1.evaluate(t)))
# plt.show()

# plt.plot(t, np.real(comp2.evaluate(t)))
# plt.plot(t, np.imag(comp2.evaluate(t)))
# plt.show()






####
#
# Signal Generator
#
####

awg = AWG()

awg.string = "AWG"
awg.comment = "AWG Device"
awg.res = 1e9
awg.t_start = 0
awg.t_end = 150e-9

ressources = [comp1, comp2]

awg.feed_ressources(ressources)



print(awg.ts)
print(awg.get_I())

plt.plot(awg.ts, awg.get_I())
plt.plot(awg.ts, awg.get_Q())
plt.show()

####
#
# CONTROL
#
####


control = Ctrl()
control.t_start = 0
control.t_end = 150e-9



# carr = Comp(
    # desc = "carrier",
    # params = carrier_parameters
# )
# print("carr uuid: " + str(carr.get_uuid()))


# comps = []
# comps.append(carr)
# comps.append(p1)
# comps.append(p2)



# sig = IQ()
# sig.t_start = 0
# sig.t_end = 150e-9
# sig.res = 1e9

# sig.calc_slice_num()
# sig.create_ts()

# sig.comps = comps


# sig.plot_IQ_components()

# sig.plot_fft_IQ_components()

# sig.plot_signal()

# sig.plot_fft_signal()


# print(sig.get_parameters())
# print(" ")
# print(" ")
# print(" ")

# print(sig.get_history())
# print(" ")
# print(" ")
# print(" ")


# sig.save_params_to_history("initial")

# print(sig.get_history())
# print(" ")
# print(" ")
# print(" ")


# sig.save_params_to_history("test2")

# print(sig.get_history())
