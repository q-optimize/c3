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

import time
import itertools
import c3po.libraries.fidelities as fidelities
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow_probability as tfp
import c3po.utils.qt_utils as qt_utils
import c3po.utils.tf_utils as tf_utils

shot_noise = True
lindblad = True
sample_num = 18

RB_length = 100
RB_number = 25
shots = 1000

def create_experiment():
    qubit_lvls = 3
    freq = 5.11722e9 * 2 * np.pi
    anhar = -315.28e6 * 2 * np.pi
    t1 = 52e-6
    t2star = 80e-6
    init_temp = 0.0
    meas_offset = 0.0
    meas_scale = 1.0
    buffer_time = 0e-9
    sim_res = 100e9
    awg_res = 2.4e9
    v2hz = 190 * 1e6 * 2 * np.pi
    t_final = sample_num / awg_res

    inphase = [0.5 * np.pi / t_final / v2hz] * sample_num
    quadrature = [0] * sample_num

    # ### MAKE MODEL
    q1 = chip.Qubit(
        name="Q1",
        desc="Qubit 1",
        comment="The one and only qubit in this chip",
        freq=Qty(
            value=freq,
            min=5.1e9 * 2 * np.pi,
            max=5.2e9 * 2 * np.pi,
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
            min=5e-6,
            max=90e-6,
            unit='s'
        ),
        t2star=Qty(
            value=t2star,
            min=10e-6,
            max=90e-6,
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
            min=0.8 * v2hz,
            max=1.2 * v2hz,
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
    generator.devices['awg'].options = 'pwc'

    # ### MAKE GATESET
    gateset = gates.GateSet()
    pwc_params = {
        'inphase': Qty(
            value=inphase,
            min=[-20 * np.pi / t_final / v2hz] * sample_num,
            max=[20 * np.pi / t_final / v2hz] * sample_num
        ),
        'quadrature': Qty(
            value=quadrature,
            min=[-20 * np.pi / t_final / v2hz] * sample_num,
            max=[20 * np.pi / t_final / v2hz] * sample_num
        ),
        'xy_angle': Qty(
            value=0.0,
            min=-0.5 * np.pi,
            max=2.5 * np.pi,
            unit='rad'
        )
    }
    pwc_env = pulse.Envelope(
        name="pwc",
        desc="PWC comp 1 of signal 1",
        params=pwc_params,
        shape=None
    )
    carrier_parameters = {
        'freq': Qty(
            value=freq,
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
    X90p.add_component(pwc_env, "d1")
    X90p.add_component(carr, "d1")
    gateset.add_instruction(X90p)

    Y90p = copy.deepcopy(X90p)
    Y90p.name = "Y90p"
    X90m = copy.deepcopy(X90p)
    X90m.name = "X90m"
    Y90m = copy.deepcopy(X90p)
    Y90m.name = "Y90m"
    Y90p.comps['d1']['pwc'].params['xy_angle'].set_value(0.5 * np.pi)
    X90m.comps['d1']['pwc'].params['xy_angle'].set_value(np.pi)
    Y90m.comps['d1']['pwc'].params['xy_angle'].set_value(1.5 * np.pi)
    gateset.add_instruction(X90m)
    gateset.add_instruction(Y90m)
    gateset.add_instruction(Y90p)

    # ### MAKE EXPERIMENT
    exp_right = Exp(model=model, generator=generator, gateset=gateset)
    return exp_right

exp_right = create_experiment()

opt_map = [
    [
      ("X90p", "d1", "pwc", "inphase"),
      ("Y90p", "d1", "pwc", "inphase"),
      ("X90m", "d1", "pwc", "inphase"),
      ("Y90m", "d1", "pwc", "inphase")
    ],
    [
      ("X90p", "d1", "pwc", "quadrature"),
      ("Y90p", "d1", "pwc", "quadrature"),
      ("X90m", "d1", "pwc", "quadrature"),
      ("Y90m", "d1", "pwc", "quadrature")
    ],
    [
      ("X90p", "d1", "carrier", "framechange"),
      ("Y90p", "d1", "carrier", "framechange"),
      ("X90m", "d1", "carrier", "framechange"),
      ("Y90m", "d1", "carrier", "framechange")
    ]
]

def ORBIT(params):
    seqs = qt_utils.single_length_RB(RB_number=RB_number, RB_length=RB_length)

    exp_right.gateset.set_parameters(params, opt_map, scaled=False)
    exp_right.opt_gates = list(
        set(itertools.chain.from_iterable(seqs))
    )
    U_dict = exp_right.get_gates()
    exp_right.evaluate(seqs)
    pop1s = exp_right.process()
    results = []

    if do_noise:
        for p1 in pop1s:
            binom = tfp.distributions.Binomial(
                total_count=shots, probs = p1
            )
            results.append(binom.sample()/shots)
    else:
        results = pop1s
    goal = np.mean(results)
    return goal
