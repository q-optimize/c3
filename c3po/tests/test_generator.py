
from c3po.control.envelopes import *
from c3po.cobj.component import ControlComponent as CtrlComp
from c3po.cobj.group import ComponentGroup as CompGroup
from c3po.control.control import Control as Control
from c3po.control.control import ControlSet as ControlSet

from c3po.control.generator import Device as Device
from c3po.control.generator import AWG as AWG
from c3po.control.generator import Mixer as Mixer
from c3po.control.generator import Generator as Generator


import uuid
import tensorflow as tf
import tensorflow_probability as tfp
import matplotlib.pyplot as plt



env_group = CompGroup()
env_group.name = "env_group"
env_group.desc = "group containing all components of type envelop"


carr_group = CompGroup()
carr_group.name = "carr_group"
carr_group.desc = "group containing all components of type carrier"



flattop_params1 = {
    'amp' : 15e6 * 2 * np.pi,
    'T_up' : 3e-9,
    'T_down' : 5e-9,
    'xy_angle' : 0.0,
    'freq_offset' : 0e6 * 2 * np.pi
}

flattop_params2 = {
    'amp' : 3e6 * 2 * np.pi,
    'T_up' : 5e-9,
    'T_down' : 7e-9,
    'xy_angle' : np.pi / 2.0,
    'freq_offset' : 0e6 * 2 * np.pi
}

params_bounds = {
    'amp' : [1e3 * 2 * np.pi, 15e6 * 2 * np.pi],
    'T_up' : [2e-9, 8e-9],
    'T_down' : [2e-9, 8e-9],
    'xy_angle' : [-np.pi, np.pi],
    'freq_offset' : [-1e9 * 2 * np.pi, 1e9 * 2 * np.pi]
}


def my_flattop(t, params):
    t_up = tf.cast(params['T_up'], tf.float64)
    t_down = tf.cast(params['T_down'], tf.float64)
    T2 = tf.maximum(t_up, t_down)
    T1 = tf.minimum(t_up, t_down)
    return (1 + tf.math.erf((t - T1) / 2e-9)) / 2 * \
(1 + tf.math.erf((-t + T2) / 2e-9)) / 2


p1 = CtrlComp(
    name = "pulse1",
    desc = "flattop comp 1 of signal 1",
    shape = my_flattop,
    params = flattop_params1,
    bounds = params_bounds,
    groups = [env_group.get_uuid()]
)
env_group.add_element(p1)


p2 = CtrlComp(
    name = "pulse2",
    desc = "flattop comp 2 of signal 1",
    shape = my_flattop,
    params = flattop_params2,
    bounds = params_bounds,
    groups = [env_group.get_uuid()]
)
env_group.add_element(p2)


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
    groups = [carr_group.get_uuid()]
)
carr_group.add_element(carr)

comps = []
comps.append(carr)
comps.append(p1)
comps.append(p2)



ctrl = Control()
ctrl.name = "control1"
ctrl.t_start = 0.0
ctrl.t_end = 10e-9
ctrl.comps = comps


ctrls = ControlSet([ctrl])


class ControlSetup(Generator):

    def __init__(
            self,
            devices = {},
            resolutions = {},
            resources = [],
            resource_groups = {}
           ):

        super().__init__(devices, resolutions, resources, resource_groups)


    def generate_signals(self, resources = []):

        if resources == []:
            resources = self.resources

        output = {}

        awg = self.devices["awg"]
        mixer = self.devices["mixer"]

        for ctrl in resources:

            awg.t_start = ctrl.t_start
            awg.t_end = ctrl.t_end
            awg.resolutions = self.resolutions
            awg.resources = [ctrl]
            awg.resource_groups = self.resource_groups
            awg.create_IQ("awg")

            #awg.plot_IQ_components("awg")
            #awg.plot_fft_IQ_components("awg")

            mixer.t_start = ctrl.t_start
            mixer.t_end = ctrl.t_end
            mixer.resolutions = self.resolutions
            mixer.resources = [ctrl]
            mixer.resource_groups = self.resource_groups
            mixer.calc_slice_num("sim")
            mixer.create_ts("sim")

            I = tfp.math.interp_regular_1d_grid(
                mixer.ts,
                x_ref_min = awg.ts[0],
                x_ref_max = awg.ts[-1],
                y_ref = awg.get_I()
                )
            Q =  tfp.math.interp_regular_1d_grid(
                mixer.ts,
                x_ref_min = awg.ts[0],
                x_ref_max = awg.ts[-1],
                y_ref = awg.get_Q()
                )

            mixer.Inphase = I
            mixer.Quadrature = Q
            mixer.combine("sim")

            output[(ctrl.name,ctrl.get_uuid())] = {"ts" : mixer.ts}
            output[(ctrl.name,ctrl.get_uuid())].update({"signal" : mixer.output})

            self.output = output

        return output



awg = AWG()
mixer = Mixer()


devices = {
    "awg" : awg,
    "mixer" : mixer
}


resolutions = {
    "awg" : 1e9,
    "sim" : 1e12
}


resources = [ctrl]


resource_groups = {
    "env" : env_group,
    "carr" : carr_group
}


gen = ControlSetup()
gen.devices = devices
gen.resolutions = resolutions
gen.resources = resources
gen.resource_groups = resource_groups


output = gen.generate_signals()


# gen.plot_signals()
# gen.plot_fft_signals()atom://teletype/portal/229ec65b-17fe-42ae-8787-3045c994f73c

# gen.plot_signals(resources)#sess = tf_debug.LocalCLIDebugWrapperSession(sess) # Enable this to debug
# gen.plot_fft_signals(resources)


# print(output)


# ts = output[(ctrl.name, ctrl.get_uuid())]["ts"]
# values = output[(ctrl.name, ctrl.get_uuid())]["signal"]


# plt.plot(ts, values)
# plt.show()
