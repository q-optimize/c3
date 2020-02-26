import copy
import numpy as np
from c3po.system.model import Model as Mdl
from c3po.c3objs import Quantity as Qty
from c3po.experiment import Experiment as Exp
from c3po.generator.generator import Generator as Gnr
import c3po.signal.gates as gates
import c3po.system.chip as chip
import c3po.generator.devices as devices
import c3po.libraries.hamiltonians as hamiltonians
import c3po.signal.pulse as pulse
import c3po.libraries.envelopes as envelopes
import c3po.system.tasks as tasks

lo_freq = 5.21e9 * 2 * np.pi
t_final = 4e-9
buffer_time = 2e-9
sim_res = 100e9
awg_res = 2e9

lindblad = True
qubit_lvls = 3
rise_time = 0.3e-9  # exact because anyway it doesn't have a big effect
v2hz = 1.0e9  # WRONG by 0.01Ghz
freq = 5.21e9 * 2 * np.pi  # WRONG FREQ by 10MHz
anhar = -305e6 * 2 * np.pi  # WRONG ANHAR by 5Mhz
t1 = 25e-6
t2star = 50e-6
init_temp = 0.05
meas_offset = -0.02
meas_scale = 1.01
p_meas_0_as_0 = 1.0
p_meas_1_as_0 = 0.01  # right
p_meas_2_as_0 = 0.0


def create_experiment():
    # ### MAKE MODEL
    q1 = chip.Qubit(
        name="Q1",
        desc="Qubit 1",
        comment="The one and only qubit in this chip",
        hilbert_dim=qubit_lvls,
        freq=Qty(
            value=freq,
            min=5.18e9 * 2 * np.pi,
            max=5.22e9 * 2 * np.pi,
            unit='rad'
        ),
        anhar=Qty(
            value=anhar,
            min=-310e6 * 2 * np.pi,
            max=-290e6 * 2 * np.pi,
            unit='rad'
        ),
        t1=Qty(
            value=t1,
            min=15e-6,
            max=35e-6,
            unit='s'
        ),
        t2star=Qty(
            value=t2star,
            min=40e-6,
            max=60e-6,
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

    init_ground = tasks.InitialiseGround(
        init_temp=Qty(
            value=init_temp,
            min=0.0,
            max=0.12,
            unit='K'
        )
    )
    meas_rescale = tasks.MeasurementRescale(
        meas_offset=Qty(
            value=meas_offset,
            min=-0.1,
            max=0.05
        ),
        meas_scale=Qty(
            value=meas_scale,
            min=0.9,
            max=1.2
        )
    )
    if p_meas_1_as_0:
        conf_matrix = tasks.ConfusionMatrix(
            confusion_row=Qty(
                value=[p_meas_0_as_0, p_meas_1_as_0, p_meas_2_as_0],
                min=[0.95, 0.0, 0.0],
                max=[1.0, 0.5, 0.5]
            )
        )
        task_list = [conf_matrix, init_ground, meas_rescale]
    else:
        task_list = [init_ground, meas_rescale]
    model = Mdl(phys_components, line_components, task_list)
    model.set_lindbladian(lindblad)

    # ### MAKE GENERATOR
    lo = devices.LO(name='lo', resolution=sim_res)
    awg = devices.AWG(name='awg', resolution=awg_res)
    mixer = devices.Mixer(name='mixer')
    v_to_hz = devices.Volts_to_Hertz(
        name='v_to_hz',
        V_to_Hz=Qty(
            value=v2hz,
            min=0.8e9,
            max=3e9,
            unit='rad/V'
        )
    )
    dig_to_an = devices.Digital_to_Analog(
        name="dac",
        resolution=sim_res
    )
    if rise_time:
        resp = devices.Response(
            name='resp',
            rise_time=Qty(
                value=rise_time,
                min=0.01e-9,
                max=1e-9,
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
            value=0.5 * np.pi * 1e-9,
            min=0.3 * np.pi * 1e-9,
            max=0.7 * np.pi * 1e-9,
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
    exp = Exp(model=model, generator=generator, gateset=gateset)
    return exp

# this comment allows me to collapse the def above
