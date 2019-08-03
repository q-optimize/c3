from c3po.signals.envelopes import *
from c3po.signals.component import Component as Comp
from c3po.signals.signal import IQ as IQ

from c3po.optimizer.optimizer import Optimizer as Optimizer

import matplotlib.pyplot as plt





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




p1 = Comp(desc = "pulse1", shape = flattop, params = flattop_params1, bounds = params_bounds)
print("p1 id: " + str(p1.get_id()))

p2 = Comp(desc = "pulse2", shape = flattop, params = flattop_params2, bounds = params_bounds)
print("p2 id: " + str(p2.get_id()))

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

carr = Comp(desc = "carrier", params = carrier_parameters, bounds = carrier_bounds)
print("carr id: " + str(carr.get_id()))


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
    'T_up' : [1,2],
    'T_down' : [1,2],
    'freq' : [3]
}

####
#
# Maybe rather simplify the structure of opt_params to the below version
#
####

# opt_params = {
    # 'T_up' : {
        # 1 : None,
        # 2 : None
        # },
    # 'T_down' : {
        # 1 : None,
        # 2 : None
        # },
    # 'freq' : {
        # 3 : None
    # }
# }



optim = Optimizer()


# print("Signal Parameter Values")
# print(sig.get_parameters())
# print(" ")
# print(" ")
# print(" ")


opt_params = optim.get_corresponding_signal_parameters(sig, opt_map)

print(opt_params)
print(" ")
print(" ")
print(" ")


opt_params['values'] = [0, 0, 0, 0, 0]
opt_params['bounds'] = [[0,0], [0,0], [0,0], [0,0], [0.0]]
print(opt_params)
print(" ")
print(" ")
print(" ")


optim.set_corresponding_signal_parameters(sig, opt_params)

opt_params = optim.get_corresponding_signal_parameters(sig, opt_map)

print(opt_params)
print(" ")
print(" ")
print(" ")



print("Signal Parameter Values")
print(sig.get_parameters())


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








