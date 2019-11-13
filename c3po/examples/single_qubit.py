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
def create_gates(t_final, v_hz_conversion, qubit_freq, qubit_anhar=None):
    """
    Define the atomic gates.

    Parameters
    ----------
    t_final : type
        Total simulation time == longest possible gate time.
    v_hz_conversion : type
        Constant relating control voltage to energy in Hertz.
    qubit_freq : type
        Qubit frequency. Determines the carrier frequency.
    qubit_anhar : type
        Qubit anharmonicity. DRAG is used if this is given.

    """
    gauss_params = {
        'amp': 0.5 * np.pi / v_hz_conversion,
        't_final': t_final,
        'xy_angle': 0.0,
        'freq_offset': 0e6 * 2 * np.pi,
        'delta': 1 / qubit_anhar
    }
    gauss_bounds = {
        'amp': [0.01 * np.pi / v_hz_conversion, 1.5 * np.pi / v_hz_conversion],
        't_final': [1e-9, 30e-9],
        'xy_angle': [-1 * np.pi/2, 1 * np.pi/2],
        'freq_offset': [-100 * 1e6 * 2 * np.pi, 100 * 1e6 * 2 * np.pi],
        'delta': [10/qubit_anhar, 0.1/qubit_anhar]
    }

    gauss_env = control.Envelope(
        name="gauss",
        desc="Gaussian comp 1 of signal 1",
        params=gauss_params,
        bounds=gauss_bounds,
        shape=envelopes.gaussian
    )
    carrier_parameters = {
        'freq': qubit_freq
    }
    carrier_bounds = {
        'freq': [4e9 * 2 * np.pi, 7e9 * 2 * np.pi]
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

    # Y90p = copy.deepcopy(X90p)
    # Y90p.name = "Y90p"
    # Y90p.comps['d1']['gauss'].params['xy_angle'] = np.pi / 2
    # Y90p.comps['d1']['gauss'].bounds['xy_angle'] = [0 * np.pi/2, 2 * np.pi/2]
    #
    # X90m = copy.deepcopy(X90p)
    # X90m.name = "X90m"
    # X90m.comps['d1']['gauss'].params['xy_angle'] = np.pi
    # X90m.comps['d1']['gauss'].bounds['xy_angle'] = [1 * np.pi/2, 3 * np.pi/2]
    #
    # Y90m = copy.deepcopy(X90p)
    # Y90m.name = "Y90m"
    # Y90m.comps['d1']['gauss'].params['xy_angle'] = - np.pi / 2
    # Y90m.comps['d1']['gauss'].bounds['xy_angle'] = [-2 * np.pi/2, 0 * np.pi/2]
    #
    # gates.add_instruction(X90m)
    # gates.add_instruction(Y90m)
    # gates.add_instruction(Y90p)
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
def create_generator(sim_res, awg_res, v_hz_conversion, logdir):
    lo = generator.LO(resolution=sim_res)
    awg = generator.AWG(resolution=awg_res, logdir=logdir)
    mixer = generator.Mixer()
    v_to_hz = generator.Volts_to_Hertz(V_to_Hz=v_hz_conversion)
    dig_to_an = generator.Digital_to_Analog(resolution=sim_res)
    resp = generator.Response(rise_time=0.1e-9, resolution=sim_res)
    # TODO Add devices by their names
    devices = [lo, awg, mixer, v_to_hz, dig_to_an, resp]
    gen = generator.Generator(devices)
    return gen
