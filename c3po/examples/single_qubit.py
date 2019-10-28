"""Creating the 1 qubit 1 drive example."""

import c3po.envelopes as envelopes
import c3po.control as control
import numpy as np

import c3po.component as component
from c3po.model import Model as Mdl

import c3po.generator as generator


# Gates
def create_gates(t_final):
    gauss_params = {
        'amp': np.pi,
        't_final': t_final,
        'xy_angle': 0.0,
        'freq_offset': 0e6 * 2 * np.pi
    }
    gauss_bounds = {
        'amp': [0.01 * np.pi, 1.5 * np.pi],
        't_final': [7e-9, 12e-9],
        'xy_angle': [-1 * np.pi/2, 1 * np.pi/2],
        'freq_offset': [-100 * 1e6 * 2 * np.pi, 100 * 1e6 * 2 * np.pi]
    }
    gauss_env = control.Envelope(
        name="gauss",
        desc="Gaussian comp 1 of signal 1",
        params=gauss_params,
        bounds=gauss_bounds,
        shape=envelopes.gaussian
    )
    carrier_parameters = {
        'freq': 5.95e9 * 2 * np.pi
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
    ctrl = control.Instruction(
        name="X90p",
        t_start=0.0,
        t_end=t_final,
        channels=["d1"]
    )
    ctrl.add_component(gauss_env, "d1")
    ctrl.add_component(carr, "d1")
    gates = control.GateSet()
    gates.add_instruction(ctrl)
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
    lo = generator.LO(resolution=sim_res)
    awg = generator.AWG(resolution=awg_res)
    mixer = generator.Mixer()
    v_to_hz = generator.Volts_to_Hertz(V_to_Hz=v_hz_conversion)
    devices = {
        "lo": lo,
        "awg": awg,
        "mixer": mixer,
        "v_to_hz": v_to_hz
        }
    gen = generator.Generator(devices)
    return gen
