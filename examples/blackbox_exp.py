import copy
import numpy as np
from c3.system.model import Model as Mdl
from c3.c3objs import Quantity as Qty
from c3.experiment import Experiment as Exp
from c3.generator.generator import Generator as Gnr
import c3.signal.gates as gates
import c3.system.chip as chip
import c3.generator.devices as devices
import c3.libraries.hamiltonians as hamiltonians
import c3.signal.pulse as pulse
import c3.libraries.envelopes as envelopes
import c3.system.tasks as tasks


def create_experiment():
    lindblad = False
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
    init_temp = 0
    qubit_temp = 0
    t_final = 7e-9  # Time for single qubit gates
    sim_res = 100e9
    awg_res = 2e9
    sideband = 50e6 * 2 * np.pi
    lo_freq_q1 = 5e9 * 2 * np.pi + sideband
    lo_freq_q2 = 5.6e9 * 2 * np.pi + sideband

    # ### MAKE MODEL
    q1 = chip.Qubit(
        name="Q1",
        desc="Qubit 1",
        freq=Qty(
            value=freq_q1,
            min_val=4.995e9 * 2 * np.pi,
            max_val=5.005e9 * 2 * np.pi,
            unit="Hz 2pi",
        ),
        anhar=Qty(
            value=anhar_q1,
            min_val=-380e6 * 2 * np.pi,
            max_val=-120e6 * 2 * np.pi,
            unit="Hz 2pi",
        ),
        hilbert_dim=qubit_lvls,
        t1=Qty(value=t1_q1, min_val=1e-6, max_val=90e-6, unit="s"),
        t2star=Qty(value=t2star_q1, min_val=10e-6, max_val=90e-6, unit="s"),
        temp=Qty(value=qubit_temp, min_val=0.0, max_val=0.12, unit="K"),
    )
    q2 = chip.Qubit(
        name="Q2",
        desc="Qubit 2",
        freq=Qty(
            value=freq_q2,
            min_val=5.595e9 * 2 * np.pi,
            max_val=5.605e9 * 2 * np.pi,
            unit="Hz 2pi",
        ),
        anhar=Qty(
            value=anhar_q2,
            min_val=-380e6 * 2 * np.pi,
            max_val=-120e6 * 2 * np.pi,
            unit="Hz 2pi",
        ),
        hilbert_dim=qubit_lvls,
        t1=Qty(value=t1_q2, min_val=1e-6, max_val=90e-6, unit="s"),
        t2star=Qty(value=t2star_q2, min_val=10e-6, max_val=90e-6, unit="s"),
        temp=Qty(value=qubit_temp, min_val=0.0, max_val=0.12, unit="K"),
    )

    q1q2 = chip.Coupling(
        name="Q1-Q2",
        desc="coupling",
        comment="Coupling qubit 1 to qubit 2",
        connected=["Q1", "Q2"],
        strength=Qty(
            value=coupling_strength,
            min_val=-1 * 1e3 * 2 * np.pi,
            max_val=200e6 * 2 * np.pi,
            unit="Hz 2pi",
        ),
        hamiltonian_func=hamiltonians.int_XX,
    )

    drive = chip.Drive(
        name="d1",
        desc="Drive 1",
        comment="Drive line 1 on qubit 1",
        connected=["Q1"],
        hamiltonian_func=hamiltonians.x_drive,
    )
    drive2 = chip.Drive(
        name="d2",
        desc="Drive 2",
        comment="Drive line 2 on qubit 2",
        connected=["Q2"],
        hamiltonian_func=hamiltonians.x_drive,
    )
    phys_components = [q1, q2]
    line_components = [drive, drive2, q1q2]

    init_ground = tasks.InitialiseGround(
        init_temp=Qty(value=init_temp, min_val=-0.001, max_val=0.22, unit="K")
    )
    task_list = [init_ground]
    model = Mdl(phys_components, line_components, task_list)
    model.set_lindbladian(lindblad)
    model.set_dressed(dressed)

    # ### MAKE GENERATOR
    generator = Gnr(
        devices={
            "LO": devices.LO(name="lo", resolution=sim_res, outputs=1),
            "AWG": devices.AWG(name="awg", resolution=awg_res, outputs=1),
            "DigitalToAnalog": devices.DigitalToAnalog(
                name="dac", resolution=sim_res, inputs=1, outputs=1
            ),
            "Response": devices.Response(
                name="resp",
                rise_time=Qty(value=0.3e-9, min_val=0.05e-9, max_val=0.6e-9, unit="s"),
                resolution=sim_res,
                inputs=1,
                outputs=1,
            ),
            "Mixer": devices.Mixer(name="mixer", inputs=2, outputs=1),
            "VoltsToHertz": devices.VoltsToHertz(
                name="v_to_hz",
                V_to_Hz=Qty(value=1e9, min_val=0.9e9, max_val=1.1e9, unit="Hz/V"),
                inputs=1,
                outputs=1,
            ),
        },
        chains={
            "d1": ["LO", "AWG", "DigitalToAnalog", "Response", "Mixer", "VoltsToHertz"],
            "d2": ["LO", "AWG", "DigitalToAnalog", "Response", "Mixer", "VoltsToHertz"],
        },
    )
    generator.devices["awg"].enable_drag_2()

    # ### MAKE GATESET
    gateset = gates.GateSet()
    gauss_params_single = {
        "amp": Qty(value=0.45, min_val=0.4, max_val=0.6, unit="V"),
        "t_final": Qty(
            value=t_final, min_val=0.5 * t_final, max_val=1.5 * t_final, unit="s"
        ),
        "sigma": Qty(
            value=t_final / 4, min_val=t_final / 8, max_val=t_final / 2, unit="s"
        ),
        "xy_angle": Qty(
            value=0.0, min_val=-0.5 * np.pi, max_val=2.5 * np.pi, unit="rad"
        ),
        "freq_offset": Qty(
            value=-sideband - 0.5e6 * 2 * np.pi,
            min_val=-53 * 1e6 * 2 * np.pi,
            max_val=-47 * 1e6 * 2 * np.pi,
            unit="Hz 2pi",
        ),
        "delta": Qty(value=-1, min_val=-5, max_val=3, unit=""),
    }

    gauss_env_single = pulse.Envelope(
        name="gauss",
        desc="Gaussian comp for single-qubit gates",
        params=gauss_params_single,
        shape=envelopes.gaussian_nonorm,
    )
    nodrive_env = pulse.Envelope(
        name="no_drive",
        params={
            "t_final": Qty(
                value=t_final, min_val=0.5 * t_final, max_val=1.5 * t_final, unit="s"
            )
        },
        shape=envelopes.no_drive,
    )
    carrier_parameters = {
        "freq": Qty(
            value=lo_freq_q1,
            min_val=4.5e9 * 2 * np.pi,
            max_val=6e9 * 2 * np.pi,
            unit="Hz 2pi",
        ),
        "framechange": Qty(value=0.0, min_val=-np.pi, max_val=3 * np.pi, unit="rad"),
    }
    carr = pulse.Carrier(
        name="carrier",
        desc="Frequency of the local oscillator",
        params=carrier_parameters,
    )
    carr_2 = copy.deepcopy(carr)
    carr_2.params["freq"].set_value(lo_freq_q2)

    RX90p_q1 = gates.Instruction(
        name="RX90p", t_start=0.0, t_end=t_final, channels=["d1"]
    )
    RX90p_q2 = gates.Instruction(
        name="RX90p", t_start=0.0, t_end=t_final, channels=["d2"]
    )
    QId_q1 = gates.Instruction(name="Id", t_start=0.0, t_end=t_final, channels=["d1"])
    QId_q2 = gates.Instruction(name="Id", t_start=0.0, t_end=t_final, channels=["d2"])

    RX90p_q1.add_component(gauss_env_single, "d1")
    RX90p_q1.add_component(carr, "d1")
    QId_q1.add_component(nodrive_env, "d1")
    QId_q1.add_component(copy.deepcopy(carr), "d1")
    QId_q1.comps["d1"]["carrier"].params["framechange"].set_value(
        (-sideband * t_final) % (2 * np.pi)
    )
    Y90p_q1 = copy.deepcopy(RX90p_q1)
    Y90p_q1.name = "Y90p"
    X90m_q1 = copy.deepcopy(RX90p_q1)
    X90m_q1.name = "X90m"
    Y90m_q1 = copy.deepcopy(RX90p_q1)
    Y90m_q1.name = "Y90m"
    Y90p_q1.comps["d1"]["gauss"].params["xy_angle"].set_value(0.5 * np.pi)
    X90m_q1.comps["d1"]["gauss"].params["xy_angle"].set_value(np.pi)
    Y90m_q1.comps["d1"]["gauss"].params["xy_angle"].set_value(1.5 * np.pi)
    Q1_gates = [QId_q1, RX90p_q1, Y90p_q1, X90m_q1, Y90m_q1]

    RX90p_q2.add_component(copy.deepcopy(gauss_env_single), "d2")
    RX90p_q2.add_component(carr_2, "d2")
    QId_q2.add_component(copy.deepcopy(nodrive_env), "d2")
    QId_q2.add_component(copy.deepcopy(carr_2), "d2")
    QId_q2.comps["d2"]["carrier"].params["framechange"].set_value(
        (-sideband * t_final) % (2 * np.pi)
    )
    Y90p_q2 = copy.deepcopy(RX90p_q2)
    Y90p_q2.name = "Y90p"
    X90m_q2 = copy.deepcopy(RX90p_q2)
    X90m_q2.name = "X90m"
    Y90m_q2 = copy.deepcopy(RX90p_q2)
    Y90m_q2.name = "Y90m"
    Y90p_q2.comps["d2"]["gauss"].params["xy_angle"].set_value(0.5 * np.pi)
    X90m_q2.comps["d2"]["gauss"].params["xy_angle"].set_value(np.pi)
    Y90m_q2.comps["d2"]["gauss"].params["xy_angle"].set_value(1.5 * np.pi)
    Q2_gates = [QId_q2, RX90p_q2, Y90p_q2, X90m_q2, Y90m_q2]

    all_1q_gates_comb = []
    for g1 in Q1_gates:
        for g2 in Q2_gates:
            g = gates.Instruction(name="NONE", t_start=0.0, t_end=t_final, channels=[])
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

    # ### MAKE EXPERIMENT
    exp = Exp(model=model, generator=generator, gateset=gateset)
    return exp
