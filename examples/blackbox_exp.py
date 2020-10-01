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

def create_experiment():
    lindblad = True
    dressed = True
    qubit_lvls = 3
    freq_q1 = 5e9 * 2 * np.pi
    freq_q2 = 5.6e9 * 2 * np.pi
    anhar_q1 = -210e6 * 2 * np.pi
    anhar_q2 = -240e6 * 2 * np.pi
    coupling_strength = 20e6 * 2 * np.pi
    t1_q1 = 27e-6
    t1_q2 = 23e-6
    t2star_q1 = 39e-6
    t2star_q2 = 31e-6
    init_temp = 50e-3
    qubit_temp = 50e-3
    m00_q1 = 0.97
    m01_q1 = 0.04
    m00_q2 = 0.96
    m01_q2 = 0.05
    v2hz = 1e9
    t_final = 7e-9   # Time for single qubit gates
    cr_time = 100e-9  # Two qubit gate
    sim_res = 100e9
    awg_res = 2e9
    meas_offset = 0.0
    meas_scale = 1.0
    sideband = 50e6 * 2 * np.pi
    lo_freq_q1 = 5e9 * 2 * np.pi + sideband
    lo_freq_q2 = 5.6e9 * 2 * np.pi + sideband

    # ### MAKE MODEL
    q1 = chip.Qubit(
        name="Q1",
        desc="Qubit 1",
        freq=Qty(
            value=freq_q1,
            min=4.995e9 * 2 * np.pi,
            max=5.005e9 * 2 * np.pi,
            unit='Hz 2pi'
        ),
        anhar=Qty(
            value=anhar_q1,
            min=-380e6 * 2 * np.pi,
            max=-120e6 * 2 * np.pi,
            unit='Hz 2pi'
        ),
        hilbert_dim=qubit_lvls,
        t1=Qty(
            value=t1_q1,
            min=1e-6,
            max=90e-6,
            unit='s'
        ),
        t2star=Qty(
            value=t2star_q1,
            min=10e-6,
            max=90e-6,
            unit='s'
        ),
        temp=Qty(
            value=qubit_temp,
            min=0.0,
            max=0.12,
            unit='K'
        )
    )
    q2 = chip.Qubit(
        name="Q2",
        desc="Qubit 2",
        freq=Qty(
            value=freq_q2,
            min=5.595e9 * 2 * np.pi,
            max=5.605e9 * 2 * np.pi,
            unit='Hz 2pi'
        ),
        anhar=Qty(
            value=anhar_q2,
            min=-380e6 * 2 * np.pi,
            max=-120e6 * 2 * np.pi,
            unit='Hz 2pi'
        ),
        hilbert_dim=qubit_lvls,
        t1=Qty(
            value=t1_q2,
            min=1e-6,
            max=90e-6,
            unit='s'
        ),
        t2star=Qty(
            value=t2star_q2,
            min=10e-6,
            max=90e-6,
            unit='s'
        ),
        temp=Qty(
            value=qubit_temp,
            min=0.0,
            max=0.12,
            unit='K'
        )
    )

    q1q2 = chip.Coupling(
        name="Q1-Q2",
        desc="coupling",
        comment="Coupling qubit 1 to qubit 2",
        connected=["Q1", "Q2"],
        strength=Qty(
            value=coupling_strength,
            min=-1 * 1e3 * 2 * np.pi,
            max=200e6 * 2 * np.pi,
            unit='Hz 2pi'
        ),
        hamiltonian_func=hamiltonians.int_XX
    )

    drive = chip.Drive(
        name="d1",
        desc="Drive 1",
        comment="Drive line 1 on qubit 1",
        connected=["Q1"],
        hamiltonian_func=hamiltonians.x_drive
    )
    drive2 = chip.Drive(
        name="d2",
        desc="Drive 2",
        comment="Drive line 2 on qubit 2",
        connected=["Q2"],
        hamiltonian_func=hamiltonians.x_drive
    )
    phys_components = [q1, q2]
    line_components = [drive, drive2, q1q2]

    one_zeros = np.array([0] * qubit_lvls)
    zero_ones = np.array([1] * qubit_lvls)
    one_zeros[0] = 1
    zero_ones[0] = 0
    val1 = one_zeros * m00_q1 + zero_ones * m01_q1
    val2 = one_zeros * m00_q2 + zero_ones * m01_q2
    min = one_zeros * 0.8 + zero_ones * 0.0
    max = one_zeros * 1.0 + zero_ones * 0.2
    confusion_row1 = Qty(value=val1, min=min, max=max, unit="")
    confusion_row2 = Qty(value=val2, min=min, max=max, unit="")
    conf_matrix = tasks.ConfusionMatrix(Q1=confusion_row1, Q2=confusion_row2)
    meas_offset = Qty(
        value=meas_offset,
        min=-0.1,
        max=0.2,
        unit=""
    )
    meas_scale = Qty(
        value=meas_scale,
        min=0.9,
        max=1.3,
        unit=""
    )
    init_ground = tasks.InitialiseGround(
        init_temp=Qty(
            value=init_temp,
            min=-0.001,
            max=0.22,
            unit='K'
        )
    )
    meas_rescale = tasks.MeasurementRescale(
        meas_offset=meas_offset,
        meas_scale=meas_scale
    )
    task_list = [conf_matrix, init_ground, meas_rescale]
    model = Mdl(phys_components, line_components, task_list)
    model.set_lindbladian(lindblad)
    model.set_dressed(dressed)

    # ### MAKE GENERATOR
    lo = devices.LO(name='lo', resolution=sim_res)
    awg = devices.AWG(name='awg', resolution=awg_res)
    mixer = devices.Mixer(name='mixer')

    v_to_hz = devices.Volts_to_Hertz(
        name='v_to_hz',
        V_to_Hz=Qty(
            value=v2hz,
            min=0.9e9,
            max=1.1e9,
            unit='Hz 2pi/V'
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
    generator.devices['awg'].options = 'IBM_drag'

    # ### MAKE GATESET
    gateset = gates.GateSet()
    gauss_params_single = {
        'amp': Qty(
            value=0.5,
            min=0.4,
            max=0.6,
            unit="V"
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
            value=-sideband - 3e6 * 2 * np.pi,
            min=-56 * 1e6 * 2 * np.pi,
            max=-52 * 1e6 * 2 * np.pi,
            unit='Hz 2pi'
        ),
        'delta': Qty(
            value=-1,
            min=-5,
            max=3,
            unit=""
        )
    }
    
    gauss_params_cr = {
        'amp': Qty(
            value=2,
            min=-5,
            max=5,
            unit="V"
        ),
        't_up': Qty(
            value=5e-9,
            min=0.0 * cr_time,
            max=0.5 * cr_time,
            unit="s"
        ),
        't_down': Qty(
            value=cr_time-5e-9,
            min=0.5 * cr_time,
            max=1.0 * cr_time,
            unit="s"
        ),
        't_final': Qty(
            value=cr_time,
            min=0.5 * cr_time,
            max=1.1 * cr_time,
            unit="s"
        ),
        'risefall': Qty(
            value=4e-9,
            min=0.0 * cr_time,
            max=1.0 * cr_time,
            unit="s"
        ),
        'xy_angle': Qty(
            value=0.0,
            min=-0.5 * np.pi,
            max=2.5 * np.pi,
            unit='pi'
        ),
        'freq_offset': Qty(
            value=-sideband,
            min=-70 * 1e6 * 2 * np.pi,
            max=-30 * 1e6 * 2 * np.pi,
            unit='Hz 2pi'
        ),
        'delta': Qty(
            value=0,
            min=-5,
            max=5,
            unit=""
        )
    }
    gauss_params_cr_2 = {
        'amp': Qty(
            value=0.032,
            min=-5,
            max=5,
            unit="V"
        ),
        't_up': Qty(
            value=5e-9,
            min=0.0 * cr_time,
            max=0.5 * cr_time,
            unit="s"
        ),
        't_down': Qty(
            value=cr_time-5e-9,
            min=0.5 * cr_time,
            max=1.0 * cr_time,
            unit="s"
        ),
        't_final': Qty(
            value=cr_time,
            min=0.5 * cr_time,
            max=1.1 * cr_time,
            unit="s"
        ),
        'risefall': Qty(
            value=4e-9,
            min=0.0 * cr_time,
            max=1.0 * cr_time,
            unit="s"
        ),
        'xy_angle': Qty(
            value=0.0,
            min=-0.5 * np.pi,
            max=2.5 * np.pi,
            unit='pi'
        ),
        'freq_offset': Qty(
            value=-sideband,
            min=-70 * 1e6 * 2 * np.pi,
            max=-30 * 1e6 * 2 * np.pi,
            unit='Hz 2pi'
        ),
        'delta': Qty(
            value=0,
            min=-5,
            max=5,
            unit=""
        )
    }        
     

    gauss_env_single = pulse.Envelope(
        name="gauss",
        desc="Gaussian comp for single-qubit gates",
        params=gauss_params_single,
        shape=envelopes.gaussian_nonorm
    )
    gauss_env_cr = pulse.Envelope(
        name="gauss",
        desc="Gaussian comp for two-qubit gates",
        params=gauss_params_cr,
        shape=envelopes.flattop_risefall
    )
    gauss_env_cr_2 = pulse.Envelope(
        name="gauss",
        desc="Gaussian comp for two-qubit gates",
        params=gauss_params_cr_2,
        shape=envelopes.flattop_risefall
    )
    nodrive_env = pulse.Envelope(
        name="no_drive",
        params={
            't_final': Qty(
                value=t_final,
                min=0.5 * t_final,
                max=1.5 * t_final,
                unit="s"
            )
        },
        shape=envelopes.no_drive
    )
    carrier_parameters = {
        'freq': Qty(
            value=lo_freq_q1,
            min=4.5e9 * 2 * np.pi,
            max=6e9 * 2 * np.pi,
            unit='Hz 2pi'
        ),
        'framechange': Qty(
            value=0.0,
            min= -np.pi,
            max= 3 * np.pi,
            unit='rad'
        )
    }
    carr = pulse.Carrier(
        name="carrier",
        desc="Frequency of the local oscillator",
        params=carrier_parameters
    )
    carr_2 = copy.deepcopy(carr)
    carr_2.params['freq'].set_value(lo_freq_q2)

    X90p_q1 = gates.Instruction(
        name="X90p",
        t_start=0.0,
        t_end=t_final,
        channels=["d1"]
    )
    X90p_q2 = gates.Instruction(
        name="X90p",
        t_start=0.0,
        t_end=t_final,
        channels=["d2"]
    )
    QId_q1 = gates.Instruction(
        name="Id",
        t_start=0.0,
        t_end=t_final,
        channels=["d1"]
    )
    QId_q2 = gates.Instruction(
        name="Id",
        t_start=0.0,
        t_end=t_final,
        channels=["d2"]
    )

    X90p_q1.add_component(gauss_env_single, "d1")
    X90p_q1.add_component(carr, "d1")
    QId_q1.add_component(nodrive_env, "d1")
    QId_q1.add_component(copy.deepcopy(carr), "d1")
    QId_q1.comps['d1']['carrier'].params['framechange'].set_value(
        (-sideband * t_final) % (2*np.pi)
    )
    Y90p_q1 = copy.deepcopy(X90p_q1)
    Y90p_q1.name = "Y90p"
    X90m_q1 = copy.deepcopy(X90p_q1)
    X90m_q1.name = "X90m"
    Y90m_q1 = copy.deepcopy(X90p_q1)
    Y90m_q1.name = "Y90m"
    Y90p_q1.comps['d1']['gauss'].params['xy_angle'].set_value(0.5 * np.pi)
    X90m_q1.comps['d1']['gauss'].params['xy_angle'].set_value(np.pi)
    Y90m_q1.comps['d1']['gauss'].params['xy_angle'].set_value(1.5 * np.pi)
    Q1_gates = [QId_q1, X90p_q1, Y90p_q1, X90m_q1, Y90m_q1]

    X90p_q2.add_component(copy.deepcopy(gauss_env_single), "d2")
    X90p_q2.add_component(carr_2, "d2")
    QId_q2.add_component(copy.deepcopy(nodrive_env), "d2")
    QId_q2.add_component(copy.deepcopy(carr_2), "d2")
    QId_q2.comps['d2']['carrier'].params['framechange'].set_value(
        (-sideband * t_final) % (2*np.pi)
    )
    Y90p_q2 = copy.deepcopy(X90p_q2)
    Y90p_q2.name = "Y90p"
    X90m_q2 = copy.deepcopy(X90p_q2)
    X90m_q2.name = "X90m"
    Y90m_q2 = copy.deepcopy(X90p_q2)
    Y90m_q2.name = "Y90m"
    Y90p_q2.comps['d2']['gauss'].params['xy_angle'].set_value(0.5 * np.pi)
    X90m_q2.comps['d2']['gauss'].params['xy_angle'].set_value(np.pi)
    Y90m_q2.comps['d2']['gauss'].params['xy_angle'].set_value(1.5 * np.pi)
    Q2_gates = [QId_q2, X90p_q2, Y90p_q2, X90m_q2, Y90m_q2]

    all_1q_gates_comb = []
    for g1 in Q1_gates:
        for g2 in Q2_gates:
            g = gates.Instruction(
                name="NONE",
                t_start=0.0,
                t_end=t_final,
                channels=[]
            )
            g.name = g1.name + ":" + g2.name
            channels = []
            channels.extend(g1.comps.keys())
            channels.extend(g2.comps.keys())
            for chan in channels:
                g.comps[chan] = {}
                if chan in g1.comps:
                    g.comps[chan].update(g1.comps[chan])
                if chan in g2.comps:
                    g.comps[chan].update(g2.comps[chan])
            all_1q_gates_comb.append(g)

    for gate in all_1q_gates_comb:
        gateset.add_instruction(gate)

    CR = gates.Instruction(
        name="CR90",
        t_start=0.0,
        t_end=cr_time,
        channels=["d1","d2"]
    )
    CR.add_component(gauss_env_cr, "d1")
    CR.add_component(carr_2, "d1")
    CR.add_component(gauss_env_cr_2, "d2")
    CR.add_component(carr_2, "d2")
    CR.comps['d1']['carrier'].params['framechange'].set_value(
            (-sideband * cr_time) % (2*np.pi)
        )
    CR.comps['d2']['carrier'].params['framechange'].set_value(
            (-sideband * cr_time) % (2*np.pi)
        )
    gateset.add_instruction(CR)

    # ### MAKE EXPERIMENT
    exp = Exp(model=model, generator=generator, gateset=gateset)
    return exp
