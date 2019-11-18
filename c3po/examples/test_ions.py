
"""Ion stuff"""

################# TODO ######################
# Set qubit levels without anharmonicity
# Appliance of drives



import c3po.hamiltonians as hamiltonians
from c3po.simulator import Simulator as Sim
import c3po.envelopes as envelopes
import c3po.control as control
import numpy as np
import copy

import c3po.component as component
from c3po.model import Model as Mdl

import pdb
import c3po.generator as generator
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.cm as cm
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
        desc="Some pi/2 pulse",
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
def create_chip_model(qubit_freq,  qubit_lvls, omega_z, res_lvls, drive_ham):
    q1 = component.Qubit(
        name="Q1",
        desc="Qubit 1",
        comment="Q1",
        freq=qubit_freq,
        #anhar=qubit_anhar,
        hilbert_dim=qubit_lvls
    )
    q2 = component.Qubit(
        name="Q2",
        desc="Qubit 2",
        comment="Q2",
        freq=qubit_freq,
        #anhar=qubit_anhar,
        hilbert_dim=qubit_lvls
    )
    r1 = component.Resonator(
        name="COM mode",
        desc="",
        comment="Centre of mass",
        hilbert_dim=res_lvls,
        freq = omega_z
    )
    r2 = component.Resonator(
        name="Stretch mode",
        desc="",
        comment="Relative motion",
        hilbert_dim=res_lvls,
        freq = omega_z*np.sqrt(3)
    )
    drive = component.Drive(
        name="D1",
        desc="Drive 1",
        comment="Drive line 1 on both qubits",
        connected=["Q1", "Q2"],
        hamiltonian=drive_ham
    )
    coupling = component.Coupling(
        name="c1",
        desc="coupl 1",
        comment="coup",
        connected=["Q1", "COM mode"],
        strength=1e9
    )
    coupling.hamiltonian=hamiltonians.int_XX
    chip_elements = [q1, q2, r1, r2, drive, coupling]
    model = Mdl(chip_elements)
    return model


# Devices and generator
def create_generator(sim_res, awg_res, v_hz_conversion, dig_to_an, logdir):
    lo = generator.LO(resolution=sim_res)
    awg = generator.AWG(resolution=awg_res, logdir=logdir)
    mixer = generator.Mixer()
    v_to_hz = generator.Volts_to_Hertz(V_to_Hz=v_hz_conversion)
    devices = {
        "lo": lo,
        "awg": awg,
        "mixer": mixer,
        "v_to_hz": v_to_hz,
        "dig_to_an": dig_to_an
    }
    gen = generator.Generator(devices)
    return gen
# System
qubit_freq = 5e9 * 2 * np.pi
qubit_anhar = -300e6 * 2 * np.pi
qubit_lvls = 3
res_lvls=4
omega_z=4e9*np.pi
drive_ham = hamiltonians.x_drive
t_final = 3e-9
v_hz_conversion = 1
digital_to_analog =1

sys_size=qubit_lvls*res_lvls

# Simulation variables
sim_res = 1e11  # 100GHz
awg_res = 1e9  # GHz

# logdir
logdir = "/home/usersFWM/susanna/python/logs/"

# Create system
model = create_chip_model(qubit_freq, qubit_lvls, omega_z, res_lvls, drive_ham)
gen = create_generator(sim_res, awg_res, v_hz_conversion, digital_to_analog, logdir)
gates = create_gates(t_final, v_hz_conversion, qubit_freq, qubit_anhar)

# Simulation class and fidelity function
sim = Sim(model, gen)#, gates)
# opt_map = gates.list_parameters()
# pulse_values, _ = gates.get_parameters(opt_map)
#model_params, _ = model.get_parameters()
model_params = model.get_parameters()
signal = gen.generate_signals(gates.instructions["X90p"])
#pdb.set_trace()
#test=Mdl([component.Qubit(name="q1")])
U = sim.propagation(signal)

Unump = U.numpy()
psi_init = np.zeros([sys_size**2,1])
psi_init[0]=1;
psi_end = np.dot(Unump, psi_init)

pdb.set_trace()
plot_psi=np.reshape(np.abs(psi_end), [sys_size, sys_size])
fig = plt.figure()
#ax = fig.gca(projection='3d')
# sys_size = d
x = np.arange(sys_size) 		# size dx1
X, Y = np.meshgrid(x, x) 		# size dxd
#x, y = X.ravel(), Y.ravel()		# size d*d,1

top = np.abs(psi_end[:,0])
#bottom = np.zeros_like(top)
#bar_width=1 # width of bar

cmap = cm.get_cmap('jet') # Get desired colormap - you can change this!
max_height = np.max(top)   # get range of colorbars so we can normalize
min_height = np.min(top)
# scale each z to [0,1], and get their rgb values
rgba = [cmap((k-min_height)/max_height) for k in top]

#ax.bar3d(x, y, bottom, bar_width, bar_width, top, shade=True, color=rgba, zsort='average')
plt.pcolor(X,Y,plot_psi, cmap="jet", vmin=min_height, vmax=max_height)
plt.colorbar()
plt.show()
