"""Creating the 1 qubit 1 drive example."""

import c3po.envelopes as envelopes
import c3po.control as control
import numpy as np
import copy

import c3po.component as component
from c3po.model import Model as Mdl
from c3po.tf_utils import tf_limit_gpu_memory as tf_limit_gpu_memory

import c3po.generator as generator

# Limit graphics memory to 1GB
tf_limit_gpu_memory(1024)


# Gates
def create_gates(t_final,
                 qubit_freq,
                 qubit_anhar,
                 amp=0.8,
                 delta=-0.5,
                 freq_offset=0.0,
                 IBM_angles=True,
                 pwc=False
                 ):

    gauss_params = {
        'amp': amp,
        't_final': t_final,
        'sigma': t_final / 4,
        'xy_angle': 0,
        'freq_offset': 0.0,
        'delta': delta,  # Delta for DRAG is defined in terms of AWG samples.
    }
    gauss_bounds = {
        'amp': [0.01, 2],
        't_final': [1e-9, 20e-9],
        'sigma': [t_final/10, t_final/2],
        'xy_angle': [-1 * np.pi/2, 1 * np.pi/2],
        'freq_offset': [-100 * 1e6, 100 * 1e6],
        'delta': [-3.0, 3.0]
    }
    gauss_env = control.Envelope(
        name="gauss",
        desc="Gaussian comp 1 of signal 1",
        params=gauss_params,
        bounds=gauss_bounds,
        shape=envelopes.gaussian_nonorm
    )
    carrier_parameters = {
        'freq': qubit_freq
    }
    carrier_bounds = {
        'freq': [4.5e9 * 2 * np.pi, 5.5e9 * 2 * np.pi]
    }
    carr = control.Carrier(
        name="carrier",
        desc="Frequency of the local oscillator",
        params=carrier_parameters,
        bounds=carrier_bounds
    )
    X90p = control.Instruction(
        name="X90p",
        t_start=0.0,
        t_end=t_final,
        channels=["d1"]
    )
    X90p.add_component(gauss_env, "d1")
    X90p.add_component(carr, "d1")

    gates = control.GateSet()
    gates.add_instruction(X90p)

    Y90p = copy.deepcopy(X90p)
    Y90p.name = "Y90p"
    Y90p.comps['d1']['gauss'].params['xy_angle'] = np.pi/2  # 1.601
    Y90p.comps['d1']['gauss'].bounds['xy_angle'] = [0 * np.pi/2, 2 * np.pi/2]

    X90m = copy.deepcopy(X90p)
    X90m.name = "X90m"
    X90m.comps['d1']['gauss'].params['xy_angle'] = np.pi  # 3.160
    X90m.comps['d1']['gauss'].bounds['xy_angle'] = [
        1 * np.pi / 2, 3 * np.pi / 2
    ]

    Y90m = copy.deepcopy(X90p)
    Y90m.name = "Y90m"
    Y90m.comps['d1']['gauss'].params['xy_angle'] = 3 * np.pi / 2  # 4.7537
    Y90m.comps['d1']['gauss'].bounds['xy_angle'] = [2 * np.pi / 2, 4 * np.pi / 2]

    if IBM_angles:
        X90p.comps['d1']['gauss'].params['xy_angle'] = 0.0399
        Y90p.comps['d1']['gauss'].params['xy_angle'] = 1.601
        X90m.comps['d1']['gauss'].params['xy_angle'] = 3.160
        Y90m.comps['d1']['gauss'].params['xy_angle'] = 4.7537

    nodrive_env = control.Envelope(
        name="nodrive_env",
        params=gauss_params,
        bounds=gauss_bounds,
        shape=envelopes.no_drive
    )
    QId = control.Instruction(
        name="QId",
        t_start=0.0,
        t_end=t_final,
        channels=["d1"]
    )
    QId.add_component(nodrive_env, "d1")
    QId.add_component(carr, "d1")

    gates.add_instruction(X90m)
    gates.add_instruction(Y90m)
    gates.add_instruction(Y90p)
    gates.add_instruction(QId)

    return gates


# Chip and model
def create_chip_model(
    qubit_freq,
    qubit_anhar,
    qubit_lvls,
    drive_ham,
    t1,
    t2star,
    temp
):
    q1 = component.Qubit(
        name="Q1",
        desc="Qubit 1",
        comment="The one and only qubit in this chip",
        freq=qubit_freq,
        anhar=qubit_anhar,
        hilbert_dim=qubit_lvls,
        # t1=t1,
        # t2star=t2star,
        # temp=temp
    )
    drive = component.Drive(
        name="D1",
        desc="Drive 1",
        comment="Drive line 1 on qubit 1",
        connected=["Q1"],
        hamiltonian=drive_ham
    )
    chip_elements = [q1, drive]
    model = Mdl(chip_elements)
    # if t1 or t2star:
    #     model.initialise_lindbladian()
    return model


# Devices and generator
def create_generator(sim_res, awg_res, v_hz_conversion, logdir):
    lo = generator.LO(name='lo', resolution=sim_res)
    awg = generator.AWG(name='awg', resolution=awg_res, logdir=logdir)
    mixer = generator.Mixer(name='mixer')
    v_to_hz = generator.Volts_to_Hertz(name='v_to_hz', V_to_Hz=v_hz_conversion)
    dig_to_an = generator.Digital_to_Analog(resolution=sim_res)
    resp = generator.Response(rise_time=0.5e-9, resolution=sim_res)

    # TODO Add devices by their names
    devices = {
        "lo": lo,
        "awg": awg,
        "mixer": mixer,
        "v_to_hz": v_to_hz,
        "dig_to_an": dig_to_an,
        "resp": resp
    }
    gen = generator.Generator(devices)
    return gen
