"""Creating the 1 qubit 1 drive example."""

import c3po.envelopes as envelopes
import c3po.control as control
import numpy as np
import copy

import c3po.component as component
from c3po.model import Model as Mdl
from c3po.component import Quantity as Qty
from c3po.tf_utils import tf_limit_gpu_memory as tf_limit_gpu_memory

import c3po.generator as generator

# Limit graphics memory to 1GB
# tf_limit_gpu_memory(1024)


# Gates
def create_gates(t_final,
                 v_hz_conversion,
                 qubit_freq,
                 qubit_anhar,
                 all_gates=True
                 ):
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
        'amp': Qty(
            value=0.5 * np.pi / v_hz_conversion,
            min=0.0 * np.pi / v_hz_conversion,
            max=1.5 * np.pi / v_hz_conversion,
            unit='V'
        ),
        't_final': t_final,
        'xy_angle': Qty(
            value=0.0,
            min=-1 * np.pi/2,
            max=1 * np.pi/2,
            unit='rad'
        ),
        'freq_offset': Qty(
            value=0e6 * 2 * np.pi,
            min=-100 * 1e6 * 2 * np.pi,
            max=100 * 1e6 * 2 * np.pi,
            unit='Hz 2pi'
        ),
        'delta': Qty(
            value=0.95 / qubit_anhar,
            min=1.5 / qubit_anhar,
            max=0.5 / qubit_anhar,
            unit='s'
        ),
    }
    gauss_env = control.Envelope(
        name="gauss",
        desc="Gaussian comp 1 of signal 1",
        params=gauss_params,
        shape=envelopes.gaussian
    )
    carrier_parameters = {
        'freq': Qty(
            value=qubit_freq.get_value(),
            min=5e9 * 2 * np.pi,
            max=5.5e9 * 2 * np.pi,
            unit='Hz 2pi'
        )
    }
    carr = control.Carrier(
        name="carrier",
        desc="Frequency of the local oscillator",
        params=carrier_parameters
    )
    X90p = control.Instruction(
        name="X90p",
        t_start=0.0,
        t_end=t_final.get_value(),
        channels=["d1"]
    )
    X90p.add_component(gauss_env, "d1")
    X90p.add_component(carr, "d1")

    gates = control.GateSet()
    gates.add_instruction(X90p)

    if all_gates:
        Y90p = copy.deepcopy(X90p)
        Y90p.name = "Y90p"
        Y90p.comps['d1']['gauss'].params['xy_angle'] = Qty(
            value=np.pi / 2,
            min=0 * np.pi/2,
            max=2 * np.pi/2,
            unit='rad'
        )

        X90m = copy.deepcopy(X90p)
        X90m.name = "X90m"
        X90m.comps['d1']['gauss'].params['xy_angle'] = Qty(
            value=np.pi,
            min=1 * np.pi/2,
            max=3 * np.pi/2,
            unit='rad'
        )

        Y90m = copy.deepcopy(X90p)
        Y90m.name = "Y90m"
        Y90m.comps['d1']['gauss'].params['xy_angle'] = Qty(
            value=-np.pi/2,
            min=-2 * np.pi/2,
            max=0 * np.pi/2,
            unit='rad'
        )
        gates.add_instruction(X90m)
        gates.add_instruction(Y90m)
        gates.add_instruction(Y90p)
    return gates


def create_pwc_gates(t_final,
                     qubit_freq,
                     inphase,
                     quadrature,
                     amp_limit,
                     all_gates=True
                     ):

    pwc_params = {
        'inphase': inphase,
        'quadrature': quadrature,
        'xy_angle': 0.0,
    }

    pwc_bounds = {
        'inphase': [-amp_limit, amp_limit] * len(inphase),
        'quadrature': [-amp_limit, amp_limit] * len(quadrature),
        'xy_angle': [-1 * np.pi/2, 1 * np.pi/2]
        }

    pwc_env = control.Envelope(
        name="pwc",
        desc="PWC comp 1 of signal 1",
        shape=envelopes.pwc,
        params=pwc_params,
        bounds=pwc_bounds,
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
    X90p.add_component(pwc_env, "d1")
    X90p.add_component(carr, "d1")

    gates = control.GateSet()
    gates.add_instruction(X90p)

    if all_gates:
        Y90p = copy.deepcopy(X90p)
        Y90p.name = "Y90p"
        Y90p.comps['d1']['pwc'].params['xy_angle'] = np.pi / 2
        Y90p.comps['d1']['pwc'].bounds['xy_angle'] = [0 * np.pi/2, 2 * np.pi/2]

        X90m = copy.deepcopy(X90p)
        X90m.name = "X90m"
        X90m.comps['d1']['pwc'].params['xy_angle'] = np.pi
        X90m.comps['d1']['pwc'].bounds['xy_angle'] = [1 * np.pi/2, 3 * np.pi/2]

        Y90m = copy.deepcopy(X90p)
        Y90m.name = "Y90m"
        Y90m.comps['d1']['pwc'].params['xy_angle'] = - np.pi / 2
        Y90m.comps['d1']['pwc'].bounds['xy_angle'] = [-2 * np.pi/2, 0 * np.pi/2]

        gates.add_instruction(X90m)
        gates.add_instruction(Y90m)
        gates.add_instruction(Y90p)
    return gates


def create_rect_gates(t_final,
                     qubit_freq,
                     amp,
                     amp_limit,
                     all_gates=True
                     ):

    rect_params = {
        'amp': amp,
        'xy_angle': 0.0,
        'freq_offset': 0e6 * 2 * np.pi
    }

    rect_bounds = {
        'amp': [-amp_limit, amp_limit],
        'xy_angle': [-1 * np.pi/2, 1 * np.pi/2],
        'freq_offset': [-100 * 1e6 * 2 * np.pi, 100 * 1e6 * 2 * np.pi]
        }

    rect_env = control.Envelope(
        name="rect",
        desc="Rectangular comp 1 of signal 1",
        shape=envelopes.rect,
        params=rect_params,
        bounds=rect_bounds,
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
    X90p.add_component(rect_env, "d1")
    X90p.add_component(carr, "d1")

    gates = control.GateSet()
    gates.add_instruction(X90p)

    if all_gates:
        Y90p = copy.deepcopy(X90p)
        Y90p.name = "Y90p"
        Y90p.comps['d1']['rect'].params['xy_angle'] = np.pi / 2
        Y90p.comps['d1']['rect'].bounds['xy_angle'] = [0 * np.pi/2, 2 * np.pi/2]

        X90m = copy.deepcopy(X90p)
        X90m.name = "X90m"
        X90m.comps['d1']['rect'].params['xy_angle'] = np.pi
        X90m.comps['d1']['rect'].bounds['xy_angle'] = [1 * np.pi/2, 3 * np.pi/2]

        Y90m = copy.deepcopy(X90p)
        Y90m.name = "Y90m"
        Y90m.comps['d1']['rect'].params['xy_angle'] = - np.pi / 2
        Y90m.comps['d1']['rect'].bounds['xy_angle'] = [-2 * np.pi/2, 0 * np.pi/2]

        gates.add_instruction(X90m)
        gates.add_instruction(Y90m)
        gates.add_instruction(Y90p)
    return gates


# Chip and model
def create_chip_model(qubit_freq, qubit_anhar, qubit_lvls, drive_ham,
                      t1=None, t2star=None, temp=None
                      ):
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
    if t1:
        q1.values['t1'] = t1
    if t2star:
        q1.values['t2star'] = t2star
    if temp:
        q1.values['temp'] = temp
    if t1 or t2star:
        model.initialise_lindbladian()
    return model


# Devices and generator
def create_generator(
    sim_res, awg_res, v_hz_conversion, logdir, rise_time=None
):
    lo = generator.LO(resolution=sim_res)
    awg = generator.AWG(resolution=awg_res, logdir=logdir)
    mixer = generator.Mixer()
    v_to_hz = generator.Volts_to_Hertz(V_to_Hz=v_hz_conversion)
    dig_to_an = generator.Digital_to_Analog(resolution=sim_res)
    resp = generator.Response(rise_time=rise_time, resolution=sim_res)
    # TODO Add devices by their names
    devices = [lo, awg, mixer, v_to_hz, dig_to_an, resp]
    gen = generator.Generator(devices)
    return gen
