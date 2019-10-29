"""Creating the 1 qubit 1 drive example."""

import c3po.envelopes as envelopes
import c3po.control as control
import numpy as np
import copy

import c3po.component as component
from c3po.model import Model as Mdl
from c3po.tf_utils import tf_limit_gpu_memory as tf_limit_gpu_memory

import c3po.generator as generator

#Limit graphics memory to 1GB
tf_limit_gpu_memory(1024)

# Gates
def create_gates(t_final, v_hz_conversion, qubit_freq, qubit_anhar):
    gauss_params = {
        'amp': 0.8,
        't_final': t_final,
        'sigma': t_final/4,
        'xy_angle': 0.0399,
        'freq_offset': 50000,
        'delta': -0.5,
    }
    gauss_bounds = {
        'amp': [0.1, 1.5],
        't_final': [7e-9, 12e-9],
        'sigma': [t_final/10, t_final/2],
        'xy_angle': [-1 * np.pi/2, 1 * np.pi/2],
        'freq_offset': [-100 * 1e6, 100 * 1e6],
        'delta': [-1.0, 0.5]
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
        'freq': [5e9 * 2 * np.pi, 7e9 * 2 * np.pi]
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

    Y90p = copy.deepcopy(X90p)
    Y90p.name = "Y90p"
    Y90p.comps['d1']['gauss'].params['xy_angle'] = 1.601
    Y90p.comps['d1']['gauss'].bounds['xy_angle'] = [0 * np.pi/2, 2 * np.pi/2]

    X90m = copy.deepcopy(X90p)
    X90m.name = "X90m"
    X90m.comps['d1']['gauss'].params['xy_angle'] = 3.160
    X90m.comps['d1']['gauss'].bounds['xy_angle'] = [1 * np.pi/2, 3 * np.pi/2]

    Y90m = copy.deepcopy(X90p)
    Y90m.name = "Y90m"
    Y90m.comps['d1']['gauss'].params['xy_angle'] = 4.7537
    Y90m.comps['d1']['gauss'].bounds['xy_angle'] = [2 * np.pi/2, 4 * np.pi/2]

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

    gates = control.GateSet()
    gates.add_instruction(X90p)
    gates.add_instruction(X90m)
    gates.add_instruction(Y90m)
    gates.add_instruction(Y90p)
    gates.add_instruction(QId)

    return gates


# Chip and model
def create_chip_model(qubit_freq, qubit_anhar, qubit_lvls, drive_ham):
    q1 = component.Qubit(
        name="Q1",
        desc="Qubit 1",
        comment="The one and only qubit in this chip",
        freq=qubit_freq,
        anhar=qubit_anhar,
        hilbert_dim=qubit_lvls
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
    return model


# Devices and generator
def create_generator(sim_res, awg_res, v_hz_conversion):
    lo = generator.LO(name='lo', resolution=sim_res)
    awg = generator.AWG(name='awg', resolution=awg_res)
    mixer = generator.Mixer(name='mixer')
    v_to_hz = generator.Volts_to_Hertz(name='v2hz', V_to_Hz=v_hz_conversion)
    devices = {
        "lo": lo,
        "awg": awg,
        "mixer": mixer,
        "v_to_hz": v_to_hz
        }
    gen = generator.Generator(devices)
    return gen
