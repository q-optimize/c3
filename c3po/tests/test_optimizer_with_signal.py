from c3po.signals.envelopes import *
from c3po.signals.component import Component as Comp
from c3po.signals.signal import IQ as IQ

from c3po.optimizer.optimizer import Optimizer as Optimizer

import matplotlib.pyplot as plt


import pprint


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


p1 = Comp(
    desc = "pulse1",
    shape = my_flattop,
    params = flattop_params1,
    bounds = params_bounds
)
print("p1 uuid: " + str(p1.get_uuid()))

p2 = Comp(
    desc = "pulse2",
    shape = my_flattop,
    params = flattop_params2,
    bounds = params_bounds
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

carrier_bounds = {
    'freq' : [2e9 * 2 * np.pi, 10e9 * 2 * np.pi]
}

carr = Comp(
    desc = "carrier",
    params = carrier_parameters,
    bounds = carrier_bounds
)
print("carr uuid: " + str(carr.get_uuid()))


comps = []
comps.append(carr)
comps.append(p1)
comps.append(p2)


sig = IQ()
sig.t_start = 0
sig.t_end = 150e-9
sig.res = 1e9

sig.calc_slice_num()
sig.create_ts()

sig.comps = comps


opt_map = {
    'T_up' : [(sig.get_uuid(), p1.get_uuid()), (sig.get_uuid(), p1.get_uuid())],
    'T_down' : [(sig.get_uuid(), p1.get_uuid()), (sig.get_uuid(), p1.get_uuid())],
    'freq' : [(sig.get_uuid(), carr.get_uuid())]
}
pprint.pprint(opt_map)
print(" ")
print(" ")
print(" ")



optim = Optimizer()


print("Signal Parameter Values")
pprint.pprint(sig.get_parameters())
print(" ")
print(" ")
print(" ")

signals = [sig]

opt_params = optim.get_corresponding_signal_parameters(signals, opt_map)

pprint.pprint(opt_params)
print(" ")
print(" ")
print(" ")


opt_params['values'] = [0, 0, 0, 0, 0]
opt_params['bounds'] = [[0,0], [0,0], [0,0], [0,0], [0.0]]
pprint.pprint(opt_params)
print(" ")
print(" ")
print(" ")


optim.set_corresponding_signal_parameters(signals, opt_params)

opt_params = optim.get_corresponding_signal_parameters(signals, opt_map)

pprint.pprint(opt_params)
print(" ")
print(" ")
print(" ")



print("Signal Parameter Values")
pprint.pprint(sig.get_parameters())


opt_settings = {

}

def evaluate_signal(signal, samples_rescaled):
    print(" ")


# optim.optimize_signal(
    # signal= sig,
    # opt_params = opt_params,
    # opt = 'cmaes',
    # settings = opt_settings,
    # calib_name = 'test',
    # eval_func = evaluate_signal
    # )
