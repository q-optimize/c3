from c3po.control.envelopes import *
from c3po.cobj.component import ControlComponent as CtrlComp
from c3po.control.control import Control as Control

from c3po.control.generator import Device as Device
from c3po.control.generator import AWG as AWG
from c3po.control.generator import Mixer as Mixer
from c3po.control.generator import Generator as Generator

import uuid
import matplotlib.pyplot as plt


comp_group = uuid.uuid4()
carrier_group = uuid.uuid4()


flattop_params1 = {
    'amp' : 15e6 * 2 * np.pi,
    'T_up' : 5e-9,
    'T_down' : 45e-9,
    'xy_angle' : 0,
    'freq_offset' : 0e6 * 2 * np.pi
}

flattop_params2 = {
    'amp' : 3e6 * 2 * np.pi,
    'T_up' : 25e-9,
    'T_down' : 30e-9,
    'xy_angle' : np.pi / 2.0,
    'freq_offset' : 0e6 * 2 * np.pi
}

params_bounds = {
    'T_up' : [2e-9, 98e-9],
    'T_down' : [2e-9, 98e-9],
    'freq_offset' : [-1e9 * 2 * np.pi, 1e9 * 2 * np.pi]
}


def my_flattop(t, params):
    t_up = params['T_up']
    t_down = params['T_down']
    return flattop(t, t_up, t_down)


p1 = CtrlComp(
    name = "pulse1",
    desc = "flattop comp 1 of signal 1",
    shape = my_flattop,
    params = flattop_params1,
    bounds = params_bounds,
    groups = [comp_group]
)
print("p1 uuid: " + str(p1.get_uuid()))

p2 = CtrlComp(
    name = "pulse2",
    desc = "flattop comp 2 of signal 1",
    shape = my_flattop,
    params = flattop_params2,
    bounds = params_bounds,
    groups = [comp_group]
)
print("p2 uuid: " + str(p2.get_uuid()))

####
# Below code: For checking the single signal components
####

# t = np.linspace(0, 150e-9, int(150e-9*1e9))
# plt.plot(t, p1.get_shape_values(t))
# plt.plot(t, p2.get_shape_values(t))
# plt.show()


carrier_parameters = {
    'freq' : 6e9 * 2 * np.pi
}

carr = CtrlComp(
    name = "carrier",
    desc = "Frequency of the local oscillator",
    params = carrier_parameters,
    groups = [carrier_group]
)
print("carr uuid: " + str(carr.get_uuid()))


comps = []
comps.append(carr)
comps.append(p1)
comps.append(p2)



ctrl = Control()
ctrl.name = "signal1"
ctrl.t_start = 0.0
ctrl.t_end = 150e-9
ctrl.comps = comps


print(ctrl.get_parameters())
print(" ")
print(" ")
print(" ")

print(ctrl.get_history())
print(" ")
print(" ")
print(" ")


ctrl.save_params_to_history("initial")

print(ctrl.get_history())
print(" ")
print(" ")
print(" ")


ctrl.save_params_to_history("test2")

print(ctrl.get_history())
