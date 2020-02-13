"""Creating the 1 qubit 1 drive example."""

import c3po.envelopes as envelopes
import c3po.control as control
import numpy as np
import copy

import c3po.component as component
from c3po.model import Model as Mdl
from c3po.component import Quantity as Qty
# from c3po.tf_utils import tf_limit_gpu_memory as tf_limit_gpu_memory

import c3po.generator as generator
import c3po.fidelities as fidelities


# Gates
def create_gates(t_final,
                 v_hz_conversion,
                 qubit_freq,
                 qubit_anhar,
                 freq_offset,
                 carrier_freq,
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
        'freq_offset': freq_offset,
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
        'freq': carrier_freq
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

        # Include identity operation
        no_drive_env = control.Envelope(
            name="no_drive",
            params=gauss_params,
            bounds=gauss_bounds,
            shape=envelopes.no_drive
        )
        Id = control.Instruction(
            name="Id",
            t_start=0.0,
            t_end=t_final,
            channels=["d1"]
        )
        Id.add_component(no_drive_env, "d1")
        Id.add_component(carr, "d1")
        gates.add_instruction(Id)

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
        Y90m.comps['d1']['pwc'].bounds['xy_angle'] = [
            -2 * np.pi/2, 0 * np.pi/2
        ]

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
        Y90p.comps['d1']['rect'].bounds['xy_angle'] = [
            0 * np.pi/2, 2 * np.pi/2
        ]

        X90m = copy.deepcopy(X90p)
        X90m.name = "X90m"
        X90m.comps['d1']['rect'].params['xy_angle'] = np.pi
        X90m.comps['d1']['rect'].bounds['xy_angle'] = [
            1 * np.pi/2, 3 * np.pi/2
        ]

        Y90m = copy.deepcopy(X90p)
        Y90m.name = "Y90m"
        Y90m.comps['d1']['rect'].params['xy_angle'] = - np.pi / 2
        Y90m.comps['d1']['rect'].bounds['xy_angle'] = [
            -2 * np.pi/2, 0 * np.pi/2
        ]

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
        hamiltonian_func=drive_ham
    )
    model = Mdl([q1], [drive])
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


def create_opt_map(pulse_type: bool, xy_angle: bool):

    # Relation functions between variables
    def add_pih(x):
        return x + 0.5 * np.pi

    def add_pi(x):
        return x + 1.0 * np.pi

    def add_mpih(x):
        return x - 0.5 * np.pi

    # Parameters to optimize
    if pulse_type == 'gauss':
        gateset_opt_map = [
            [('X90p', 'd1', 'gauss', 'amp'),
             ('Y90p', 'd1', 'gauss', 'amp'),
             ('X90m', 'd1', 'gauss', 'amp'),
             ('Y90m', 'd1', 'gauss', 'amp')],
            [('X90p', 'd1', 'gauss', 'freq_offset'),
             ('Y90p', 'd1', 'gauss', 'freq_offset'),
             ('X90m', 'd1', 'gauss', 'freq_offset'),
             ('Y90m', 'd1', 'gauss', 'freq_offset')]
        ]
        if xy_angle:
            gateset_opt_map.append(
                [('X90p', 'd1', 'gauss', 'xy_angle'),
                 ('Y90p', 'd1', 'gauss', 'xy_angle', add_pih),
                 ('X90m', 'd1', 'gauss', 'xy_angle', add_pi),
                 ('Y90m', 'd1', 'gauss', 'xy_angle', add_mpih)]
            )
    elif pulse_type == 'drag':
        gateset_opt_map = [
            [('X90p', 'd1', 'gauss', 'amp'),
             ('Y90p', 'd1', 'gauss', 'amp'),
             ('X90m', 'd1', 'gauss', 'amp'),
             ('Y90m', 'd1', 'gauss', 'amp')],
            [('X90p', 'd1', 'gauss', 'freq_offset'),
             ('Y90p', 'd1', 'gauss', 'freq_offset'),
             ('X90m', 'd1', 'gauss', 'freq_offset'),
             ('Y90m', 'd1', 'gauss', 'freq_offset')],
            [('X90p', 'd1', 'gauss', 'delta'),
             ('Y90p', 'd1', 'gauss', 'delta'),
             ('X90m', 'd1', 'gauss', 'delta'),
             ('Y90m', 'd1', 'gauss', 'delta')]
        ]
        if xy_angle:
            gateset_opt_map.append(
                [('X90p', 'd1', 'gauss', 'xy_angle'),
                 ('Y90p', 'd1', 'gauss', 'xy_angle', add_pih),
                 ('X90m', 'd1', 'gauss', 'xy_angle', add_pi),
                 ('Y90m', 'd1', 'gauss', 'xy_angle', add_mpih)]
            )
    elif pulse_type == 'pwc':
        gateset_opt_map = [
            [('X90p', 'd1', 'pwc', 'inphase'),
             ('Y90p', 'd1', 'pwc', 'inphase'),
             ('X90m', 'd1', 'pwc', 'inphase'),
             ('Y90m', 'd1', 'pwc', 'inphase')],
            [('X90p', 'd1', 'pwc', 'quadrature'),
             ('Y90p', 'd1', 'pwc', 'quadrature'),
             ('X90m', 'd1', 'pwc', 'quadrature'),
             ('Y90m', 'd1', 'pwc', 'quadrature')]
        ]

    elif pulse_type == 'rect':
        gateset_opt_map = [
            [('X90p', 'd1', 'rect', 'amp'),
             ('Y90p', 'd1', 'rect', 'amp'),
             ('X90m', 'd1', 'rect', 'amp'),
             ('Y90m', 'd1', 'rect', 'amp')]
        ]
        if xy_angle:
            gateset_opt_map.append(
                [('X90p', 'd1', 'rect', 'xy_angle'),
                 ('Y90p', 'd1', 'rect', 'xy_angle', add_pih),
                 ('X90m', 'd1', 'rect', 'xy_angle', add_pi),
                 ('Y90m', 'd1', 'rect', 'xy_angle', add_mpih)]
            )
    return gateset_opt_map


# Fidelity fucntions
def create_fcts(lindbladian,
    U_dict=True,
    RB_number = 20,
    RB_length = 50,
    shots = 500):
    # Define infidelity functions (search fucntions)'

    def store_U_dict(U_dict):
        return U_dict

    if not lindbladian:

        def unit_compsub_X90p(U_dict):
            return fidelities.unitary_infid(U_dict, 'X90p', proj=True)
        def unit_compsub_Y90p(U_dict):
            return fidelities.unitary_infid(U_dict, 'Y90p', proj=True)
        def unit_compsub_X90m(U_dict):
            return fidelities.unitary_infid(U_dict, 'X90m', proj=True)
        def unit_compsub_Y90m(U_dict):
            return fidelities.unitary_infid(U_dict, 'Y90m', proj=True)

        def unit_fulluni_X90p(U_dict):
            return fidelities.unitary_infid(U_dict, 'X90p', proj=False)
        def unit_fulluni_Y90p(U_dict):
            return fidelities.unitary_infid(U_dict, 'Y90p', proj=False)
        def unit_fulluni_X90m(U_dict):
            return fidelities.unitary_infid(U_dict, 'X90m', proj=False)
        def unit_fulluni_Y90m(U_dict):
            return fidelities.unitary_infid(U_dict, 'Y90m', proj=False)

        def avfid_compsub_X90p(U_dict):
            return fidelities.average_infid(U_dict, 'X90p', proj=True)
        def avfid_compsub_Y90p(U_dict):
            return fidelities.average_infid(U_dict, 'Y90p', proj=True)
        def avfid_compsub_X90m(U_dict):
            return fidelities.average_infid(U_dict, 'X90m', proj=True)
        def avfid_compsub_Y90m(U_dict):
            return fidelities.average_infid(U_dict, 'Y90m', proj=True)

        def avfid_fulluni_X90p(U_dict):
            return fidelities.average_infid(U_dict, 'X90p', proj=False)
        def avfid_fulluni_Y90p(U_dict):
            return fidelities.average_infid(U_dict, 'Y90p', proj=False)
        def avfid_fulluni_X90m(U_dict):
            return fidelities.average_infid(U_dict, 'X90m', proj=False)
        def avfid_fulluni_Y90m(U_dict):
            return fidelities.average_infid(U_dict, 'Y90m', proj=False)

        def epc_ana_compsub(U_dict):
            return fidelities.epc_analytical(U_dict, proj=True)
        def epc_ana_fulluni(U_dict):
            return fidelities.epc_analytical(U_dict, proj=False)

        def pop0_X90p_0_fulluni(U_dict):
            return fidelities.population(U_dict, 0, 'X90p')
        def pop0_X90p_1_fulluni(U_dict):
            return fidelities.population(U_dict, 1, 'X90p')
        def pop0_X90p_2_fulluni(U_dict):
            return fidelities.population(U_dict, 2, 'X90p')
        def pop0_X90p_3_fulluni(U_dict):
            return fidelities.population(U_dict, 3, 'X90p')

        def pop0_Y90p_2_fulluni(U_dict):
            return fidelities.population(U_dict, 2, 'Y90p')
        def pop0_X90m_2_fulluni(U_dict):
            return fidelities.population(U_dict, 2, 'X90m')
        def pop0_Y90m_2_fulluni(U_dict):
            return fidelities.population(U_dict, 2, 'Y90m')

    elif lindbladian:

        def unit_compsub_X90p(U_dict):
            return fidelities.lindbladian_unitary_infid(U_dict, 'X90p', proj=True)
        def unit_compsub_Y90p(U_dict):
            return fidelities.lindbladian_unitary_infid(U_dict, 'Y90p', proj=True)
        def unit_compsub_X90m(U_dict):
            return fidelities.lindbladian_unitary_infid(U_dict, 'X90m', proj=True)
        def unit_compsub_Y90m(U_dict):
            return fidelities.lindbladian_unitary_infid(U_dict, 'Y90m', proj=True)

        def unit_fulluni_X90p(U_dict):
            return fidelities.lindbladian_unitary_infid(U_dict, 'X90p', proj=False)
        def unit_fulluni_Y90p(U_dict):
            return fidelities.lindbladian_unitary_infid(U_dict, 'Y90p', proj=False)
        def unit_fulluni_X90m(U_dict):
            return fidelities.lindbladian_unitary_infid(U_dict, 'X90m', proj=False)
        def unit_fulluni_Y90m(U_dict):
            return fidelities.lindbladian_unitary_infid(U_dict, 'Y90m', proj=False)

        def avfid_compsub_X90p(U_dict):
            return fidelities.lindbladian_average_infid(U_dict, 'X90p', proj=True)
        def avfid_compsub_Y90p(U_dict):
            return fidelities.lindbladian_average_infid(U_dict, 'Y90p', proj=True)
        def avfid_compsub_X90m(U_dict):
            return fidelities.lindbladian_average_infid(U_dict, 'X90m', proj=True)
        def avfid_compsub_Y90m(U_dict):
            return fidelities.lindbladian_average_infid(U_dict, 'Y90m', proj=True)

        def avfid_fulluni_X90p(U_dict):
            return fidelities.lindbladian_average_infid(U_dict, 'X90p', proj=False)
        def avfid_fulluni_Y90p(U_dict):
            return fidelities.lindbladian_average_infid(U_dict, 'Y90p', proj=False)
        def avfid_fulluni_X90m(U_dict):
            return fidelities.lindbladian_average_infid(U_dict, 'X90m', proj=False)
        def avfid_fulluni_Y90m(U_dict):
            return fidelities.lindbladian_average_infid(U_dict, 'Y90m', proj=False)

        def epc_ana_compsub(U_dict):
            return fidelities.lindbladian_epc_analytical(U_dict, proj=True)
        def epc_ana_fulluni(U_dict):
            return fidelities.lindbladian_epc_analytical(U_dict, proj=False)

        def pop0_X90p_0_fulluni(U_dict):
            return fidelities.lindbladian_population(U_dict, 0, 'X90p')
        def pop0_X90p_1_fulluni(U_dict):
            return fidelities.lindbladian_population(U_dict, 1, 'X90p')
        def pop0_X90p_2_fulluni(U_dict):
            return fidelities.lindbladian_population(U_dict, 2, 'X90p')
        def pop0_X90p_3_fulluni(U_dict):
            return fidelities.lindbladian_population(U_dict, 3, 'X90p')

        def pop0_Y90p_2_fulluni(U_dict):
            return fidelities.lindbladian_population(U_dict, 2, 'Y90p')
        def pop0_X90m_2_fulluni(U_dict):
            return fidelities.lindbladian_population(U_dict, 2, 'X90m')
        def pop0_Y90m_2_fulluni(U_dict):
            return fidelities.lindbladian_population(U_dict, 2, 'Y90m')

    def epc_RB(U_dict):
        return fidelities.RB(U_dict, logspace=True, lindbladian=lindbladian)[0]
    def RB_fig(U_dict):
        return fidelities.RB(U_dict, logspace=True, lindbladian=lindbladian)[-2:]
    def epc_leakage_RB(U_dict):
        return fidelities.leakage_RB(U_dict,
            logspace=True, lindbladian=lindbladian)[0]

    seqs = single_length_RB(RB_number=RB_number, RB_length=RB_length)
    def orbit_no_noise(U_dict):
        return fidelities.orbit_infid(U_dict, lindbladian=lindbladian,
            seqs=seqs)
    def orbit_seq_noise(U_dict):
        return fidelities.orbit_infid(U_dict, lindbladian=lindbladian,
            RB_number=RB_number, RB_length=RB_length)
    def orbit_shot_noise(U_dict):
        return fidelities.orbit_infid(U_dict, lindbladian=lindbladian,
            seqs=seqs, shots=shots)
    def orbit_seq_shot_noise(U_dict):
        return fidelities.orbit_infid(U_dict,lindbladian=lindbladian,
            shots=shots, RB_number=RB_number, RB_length=RB_length)
    seqs6 = single_length_RB(RB_number=RB_number, RB_length=6)
    seqs10 = single_length_RB(RB_number=RB_number, RB_length=10)
    seqs20 = single_length_RB(RB_number=RB_number, RB_length=20)
    seqs40 = single_length_RB(RB_number=RB_number, RB_length=40)
    def orbit6(U_dict):
        return fidelities.orbit_infid(U_dict, lindbladian=lindbladian,
            seqs=seqs6)
    def orbit10(U_dict):
        return fidelities.orbit_infid(U_dict, lindbladian=lindbladian,
            seqs=seqs10)
    def orbit20(U_dict):
        return fidelities.orbit_infid(U_dict, lindbladian=lindbladian,
            seqs=seqs20)
    def orbit40(U_dict):
        return fidelities.orbit_infid(U_dict, lindbladian=lindbladian,
            seqs=seqs40)

    fcts_list = [
        unit_compsub_X90p,
        unit_compsub_Y90p,
        unit_compsub_X90m,
        unit_compsub_Y90m,
        # unit_fulluni_X90p,
        # unit_fulluni_Y90p,
        # unit_fulluni_X90m,
        # unit_fulluni_Y90m,
        avfid_compsub_X90p,
        avfid_compsub_Y90p,
        avfid_compsub_X90m,
        avfid_compsub_Y90m,
        # avfid_fulluni_X90p,
        # avfid_fulluni_Y90p,
        # avfid_fulluni_X90m,
        # avfid_fulluni_Y90m,
        epc_ana_compsub,
        epc_ana_fulluni,
        epc_RB,
        epc_leakage_RB,
        orbit6,
        orbit10,
        orbit20,
        orbit40,
        orbit_no_noise,
        orbit_seq_noise,
        orbit_shot_noise,
        orbit_seq_shot_noise,
        pop0_X90p_0_fulluni,
        pop0_X90p_1_fulluni,
        pop0_X90p_2_fulluni,
        pop0_X90p_3_fulluni,
        pop0_Y90p_2_fulluni,
        pop0_X90m_2_fulluni,
        pop0_Y90m_2_fulluni,
    ]

    if U_dict:
        fcts_list.append(store_U_dict)

    fcts_dict = {}
    for fct in fcts_list:
        name = fct.__name__
        fcts_dict[name] = fct

    return fcts_dict
