import copy
import numpy as np
from c3.model import Model as Mdl
from c3.c3objs import Quantity as Qty
from c3.parametermap import ParameterMap as PMap
from c3.experiment import Experiment as Exp
from c3.generator.generator import Generator as Gnr
import c3.signal.gates as gates
import c3.libraries.chip as chip
import c3.generator.devices as devices
import c3.libraries.hamiltonians as hamiltonians
import c3.signal.pulse as pulse
import c3.libraries.envelopes as envelopes
import c3.libraries.tasks as tasks


def create_experiment():
    lindblad = False
    dressed = True
    qubit_lvls = 3
    freq = 5e9
    anhar = -210e6
    init_temp = 0
    qubit_temp = 0
    t_final = 7e-9  # Time for single qubit gates
    sim_res = 100e9
    awg_res = 2e9
    sideband = 50e6
    lo_freq = 5e9 + sideband

    # ### MAKE MODEL
    q1 = chip.Qubit(
        name="Q1",
        desc="Qubit 1",
        freq=Qty(
            value=freq,
            min_val=4.995e9,
            max_val=5.005e9,
            unit="Hz 2pi",
        ),
        anhar=Qty(
            value=anhar,
            min_val=-380e6,
            max_val=-120e6,
            unit="Hz 2pi",
        ),
        hilbert_dim=qubit_lvls,
        temp=Qty(value=qubit_temp, min_val=0.0, max_val=0.12, unit="K"),
    )

    drive = chip.Drive(
        name="d1",
        desc="Drive 1",
        comment="Drive line 1 on qubit 1",
        connected=["Q1"],
        hamiltonian_func=hamiltonians.x_drive,
    )
    phys_components = [q1]
    line_components = [drive]

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
            "d1": ["LO", "AWG", "DigitalToAnalog", "Response", "Mixer", "VoltsToHertz"]
        },
    )
    generator.devices["AWG"].enable_drag_2()

    # ### MAKE GATESET
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
            value=-sideband - 0.5e6,
            min_val=-60 * 1e6,
            max_val=-40 * 1e6,
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
            value=lo_freq,
            min_val=4.5e9,
            max_val=6e9,
            unit="Hz 2pi",
        ),
        "framechange": Qty(value=0.0, min_val=-np.pi, max_val=3 * np.pi, unit="rad"),
    }
    carr = pulse.Carrier(
        name="carrier",
        desc="Frequency of the local oscillator",
        params=carrier_parameters,
    )

    rx90p = gates.Instruction(
        name="rx90p", t_start=0.0, t_end=t_final, channels=["d1"], targets=[0]
    )
    QId = gates.Instruction(
        name="id", t_start=0.0, t_end=t_final, channels=["d1"], targets=[0]
    )

    rx90p.add_component(gauss_env_single, "d1")
    rx90p.add_component(carr, "d1")
    QId.add_component(nodrive_env, "d1")
    QId.add_component(copy.deepcopy(carr), "d1")
    QId.comps["d1"]["carrier"].params["framechange"].set_value(
        (-sideband * t_final) % (2 * np.pi)
    )
    ry90p = copy.deepcopy(rx90p)
    ry90p.name = "ry90p"
    rx90m = copy.deepcopy(rx90p)
    rx90m.name = "rx90m"
    ry90m = copy.deepcopy(rx90p)
    ry90m.name = "ry90m"
    ry90p.comps["d1"]["gauss"].params["xy_angle"].set_value(0.5 * np.pi)
    rx90m.comps["d1"]["gauss"].params["xy_angle"].set_value(np.pi)
    ry90m.comps["d1"]["gauss"].params["xy_angle"].set_value(1.5 * np.pi)

    parameter_map = PMap(
        instructions=[QId, rx90p, ry90p, rx90m, ry90m], model=model, generator=generator
    )

    # ### MAKE EXPERIMENT
    exp = Exp(pmap=parameter_map)
    return exp
