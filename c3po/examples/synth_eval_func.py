import copy
import numpy as np
import tensorflow as tf
from c3po.system.model import Model as Mdl
from c3po.c3objs import Quantity as Qty
from c3po.experiment import Experiment as Exp
from c3po.generator.generator import Generator as Gnr
import c3po.signal.gates as gates
import c3po.system.chip as chip
import c3po.generator.devices as devices
import c3po.libraries.hamiltonians as hamiltonians
import c3po.libraries.fidelities as fidelities
import c3po.signal.pulse as pulse
import c3po.libraries.envelopes as envelopes
import c3po.system.tasks as tasks
import c3po.utils.qt_utils as qt_utils


lindblad = True
qubit_lvls = 3
freq = 5.2e9 * 2 * np.pi
lo_freq = 5.22e9 * 2 * np.pi
anhar = -300e6 * 2 * np.pi
t1 = 80e-6
init_temp = 0.05
meas_offset = -0.02
meas_scale = 1.02
t_final = 5e-9
buffer_time = 1e-9
sim_res = 60e9
awg_res = 1e9

q1 = chip.Qubit(
    name="Q1",
    desc="Qubit 1",
    comment="The one and only qubit in this chip",
    freq=Qty(
        value=freq,
        min=5.15e9 * 2 * np.pi,
        max=5.4e9 * 2 * np.pi,
        unit='rad'
    ),
    anhar=Qty(
        value=anhar,
        min=-380e6 * 2 * np.pi,
        max=-220e6 * 2 * np.pi,
        unit='rad'
    ),
    hilbert_dim=qubit_lvls,
    t1=Qty(
        value=t1,
        min=10e-6,
        max=90e-6,
        unit='s'
    ),
    t2star=Qty(
        value=25e-6,
        min=10e-6,
        max=50e-6,
        unit='s'
    ),
    temp=Qty(
        value=init_temp,
        min=0.0,
        max=0.12,
        unit='K'
    )
)
drive = chip.Drive(
    name="d1",
    desc="Drive 1",
    comment="Drive line 1 on qubit 1",
    connected=["Q1"],
    hamiltonian_func=hamiltonians.x_drive
)
phys_components = [q1]
line_components = [drive]

one_zeros = np.array([0] * qubit_lvls)
zero_ones = np.array([1] * qubit_lvls)
one_zeros[0] = 1
zero_ones[0] = 0
val = one_zeros * 1.0 + zero_ones * 0.0
min = one_zeros * 0.95 + zero_ones * 0.0
max = one_zeros * 1.0 + zero_ones * 0.05
confusion_row = Qty(value=val, min=min, max=max)
meas_offset = Qty(
    value=meas_offset,
    min=-0.1,
    max=0.05
)
meas_scale = Qty(
    value=meas_scale,
    min=0.9,
    max=1.2
)
conf_matrix = tasks.ConfusionMatrix(confusion_row=confusion_row)
init_ground = tasks.InitialiseGround(
    init_temp=Qty(
        value=init_temp,
        min=0.0,
        max=0.12,
        unit='K'
    )
)
meas_rescale = tasks.MeasurementRescale(
    meas_offset=meas_offset,
    meas_scale=meas_scale)
task_list = [conf_matrix, init_ground, meas_rescale]
model = Mdl(phys_components, line_components, task_list)
model.set_lindbladian(lindblad)

# ### MAKE GENERATOR
lo = devices.LO(name='lo', resolution=sim_res)
awg = devices.AWG(name='awg', resolution=awg_res)
mixer = devices.Mixer(name='mixer')

v_to_hz = devices.Volts_to_Hertz(
    name='v_to_hz',
    V_to_Hz=Qty(
        value=1,
        min=0.8,
        max=3,
        unit='rad/V'
    )
)
dig_to_an = devices.Digital_to_Analog(
    name="dac",
    resolution=sim_res
)
resp = devices.Response(
    name='resp',
    rise_time=Qty(
        value=0.3e-9,
        min=0.05e-9,
        max=0.6e-9,
        unit='s'
    ),
    resolution=sim_res
)

device_list = [lo, awg, mixer, v_to_hz, dig_to_an, resp]
generator = Gnr(device_list)
generator.devices['awg'].options = 'drag'

# ### MAKE GATESET
gateset = gates.GateSet()
gauss_params = {
    'amp': Qty(
        value=0.5 * np.pi,
        min=0.3 * np.pi,
        max=0.7 * np.pi,
    ),
    't_final': Qty(
        value=t_final,
        min=0.5 * t_final,
        max=1.5 * t_final,
        unit="s"
    ),
    'sigma': Qty(
        value=t_final / 4,
        min=t_final / 8,
        max=t_final / 2,
        unit="s"
    ),
    'xy_angle': Qty(
        value=0.0,
        min=-0.5 * np.pi,
        max=2.5 * np.pi,
        unit='rad'
    ),
    'freq_offset': Qty(
        value=0e6 * 2 * np.pi,
        min=-100 * 1e6 * 2 * np.pi,
        max=100 * 1e6 * 2 * np.pi,
        unit='Hz 2pi'
    ),
    'delta': Qty(
        value=0.5 / anhar,
        min=1.5 / anhar,
        max=0.0 / anhar
    ),
}
gauss_env = pulse.Envelope(
    name="gauss",
    desc="Gaussian comp 1 of signal 1",
    params=gauss_params,
    shape=envelopes.gaussian
)
carrier_parameters = {
    'freq': Qty(
        value=lo_freq,
        min=5e9 * 2 * np.pi,
        max=5.5e9 * 2 * np.pi,
        unit='Hz 2pi'
    ),
    'framechange': Qty(
        value=0.0,
        min=-np.pi,
        max=np.pi,
        unit='rad'
    )
}
carr = pulse.Carrier(
    name="carrier",
    desc="Frequency of the local oscillator",
    params=carrier_parameters
)
X90p = gates.Instruction(
    name="X90p",
    t_start=0.0,
    t_end=t_final+buffer_time,
    channels=["d1"]
)
X90p.add_component(gauss_env, "d1")
X90p.add_component(carr, "d1")
gateset.add_instruction(X90p)

# nodrive_env = pulse.Envelope(
#     name="nodrive_env",
#     params=gauss_params,
#     shape=envelopes.no_drive
# )
# QId = gates.Instruction(
#     name="QId",
#     t_start=0.0,
#     t_end=t_final+buffer_time,
#     channels=["d1"]
# )
# QId.add_component(nodrive_env, "d1")
# QId.add_component(carr, "d1")
# gateset.add_instruction(QId)

Y90p = copy.deepcopy(X90p)
Y90p.name = "Y90p"
X90m = copy.deepcopy(X90p)
X90m.name = "X90m"
Y90m = copy.deepcopy(X90p)
Y90m.name = "Y90m"
Y90p.comps['d1']['gauss'].params['xy_angle'].set_value(0.5 * np.pi)
X90m.comps['d1']['gauss'].params['xy_angle'].set_value(np.pi)
Y90m.comps['d1']['gauss'].params['xy_angle'].set_value(1.5 * np.pi)
gateset.add_instruction(X90m)
gateset.add_instruction(Y90m)
gateset.add_instruction(Y90p)

# ### MAKE EXPERIMENT
exp_right = Exp(model=model, generator=generator, gateset=gateset)


# def eval_func(params, exp_right, opt_map):
#     def unit_X90p(U_dict):
#         return fidelities.unitary_infid(U_dict, 'X90p', proj=True)
#     exp_right.gateset.set_parameters(params, opt_map, scaled=False)
#     U_dict = exp_right.get_gates()
#     goal = float(unit_X90p(U_dict).numpy())
#     seqs = [['X90p']]
#     results = [goal]
#     results_std = [0.03]
#     return goal, results, results_std, seqs

def eval_func(params, exp_right, opt_map):
    shots = 500
    RB_number = 20
    RB_length = 30
    exp_right.gateset.set_parameters(params, opt_map, scaled=False)
    seqs = qt_utils.single_length_RB(RB_number=RB_number, RB_length=RB_length)
    pop1s = exp_right.evaluate(seqs)
    results = []
    results_std = []
    for p1 in pop1s:
        draws = tf.keras.backend.random_binomial(
            [shots],
            p=p1,
            dtype=tf.float64,
        )
        results.append(np.mean(draws))
        results_std.append(np.std(draws))
    goal = np.mean(results)
    return goal, results, results_std, seqs
