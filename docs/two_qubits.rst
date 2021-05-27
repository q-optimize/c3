.. _setup-example:

Setup of a two-qubit chip with :math:`C^3`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

In this example we will set-up a two qubit quantum processor and define
a simple gate.

Imports
^^^^^^^

.. code-block:: python

    # System imports
    import copy
    import numpy as np
    import time
    import itertools
    import matplotlib.pyplot as plt
    import tensorflow as tf
    import tensorflow_probability as tfp

    # Main C3 objects
    from c3.c3objs import Quantity as Qty
    from c3.parametermap import ParameterMap as PMap
    from c3.experiment import Experiment as Exp
    from c3.model import Model as Mdl
    from c3.generator.generator import Generator as Gnr

    # Building blocks
    import c3.generator.devices as devices
    import c3.signal.gates as gates
    import c3.libraries.chip as chip
    import c3.signal.pulse as pulse
    import c3.libraries.tasks as tasks

    # Libs and helpers
    import c3.libraries.algorithms as algorithms
    import c3.libraries.hamiltonians as hamiltonians
    import c3.libraries.fidelities as fidelities
    import c3.libraries.envelopes as envelopes
    import c3.utils.qt_utils as qt_utils
    import c3.utils.tf_utils as tf_utils

Model components
^^^^^^^^^^^^^^^^

We first create a qubit. Each parameter is a Quantity (``Qty()``) object
with bounds and a unit. In :math:`C^3`, the default multi-level qubit is
a Transmon modelled as a Duffing oscillator with frequency
:math:`\omega` and anharmonicity :math:`\delta` :

.. math::

    H/\hbar = \omega b^\dagger b - \frac{\delta}{2}                        \left(b^\dagger b - 1\right) b^\dagger b

The “name” will be used to identify this qubit (or other component)
later and should thus be chosen carefully.

.. code-block:: python

    qubit_lvls = 3
    freq_q1 = 5e9
    anhar_q1 = -210e6
    t1_q1 = 27e-6
    t2star_q1 = 39e-6
    qubit_temp = 50e-3

    q1 = chip.Qubit(
        name="Q1",
        desc="Qubit 1",
        freq=Qty(
            value=freq_q1,
            min_val=4.995e9 ,
            max_val=5.005e9 ,
            unit='Hz 2pi'
        ),
        anhar=Qty(
            value=anhar_q1,
            min_val=-380e6 ,
            max_val=-120e6 ,
            unit='Hz 2pi'
        ),
        hilbert_dim=qubit_lvls,
        t1=Qty(
            value=t1_q1,
            min_val=1e-6,
            max_val=90e-6,
            unit='s'
        ),
        t2star=Qty(
            value=t2star_q1,
            min_val=10e-6,
            max_val=90e-3,
            unit='s'
        ),
        temp=Qty(
            value=qubit_temp,
            min_val=0.0,
            max_val=0.12,
            unit='K'
        )
    )

And the same for a second qubit.

.. code-block:: python

    freq_q2 = 5.6e9
    anhar_q2 = -240e6
    t1_q2 = 23e-6
    t2star_q2 = 31e-6
    q2 = chip.Qubit(
        name="Q2",
        desc="Qubit 2",
        freq=Qty(
            value=freq_q2,
            min_val=5.595e9 ,
            max_val=5.605e9 ,
            unit='Hz 2pi'
        ),
        anhar=Qty(
            value=anhar_q2,
            min_val=-380e6 ,
            max_val=-120e6 ,
            unit='Hz 2pi'
        ),
        hilbert_dim=qubit_lvls,
        t1=Qty(
            value=t1_q2,
            min_val=1e-6,
            max_val=90e-6,
            unit='s'
        ),
        t2star=Qty(
            value=t2star_q2,
            min_val=10e-6,
            max_val=90e-6,
            unit='s'
        ),
        temp=Qty(
            value=qubit_temp,
            min_val=0.0,
            max_val=0.12,
            unit='K'
        )
    )

A static coupling between the two is realized in the following way. We
supply the type of coupling by selecting ``int_XX``
:math:`(b_1+b_1^\dagger)(b_2+b_2^\dagger)` from the hamiltonian library.
The “connected” property contains the list of qubit names to be coupled,
in this case “Q1” and “Q2”.

.. code-block:: python

    coupling_strength = 20e6
    q1q2 = chip.Coupling(
        name="Q1-Q2",
        desc="coupling",
        comment="Coupling qubit 1 to qubit 2",
        connected=["Q1", "Q2"],
        strength=Qty(
            value=coupling_strength,
            min_val=-1 * 1e3 ,
            max_val=200e6 ,
            unit='Hz 2pi'
        ),
        hamiltonian_func=hamiltonians.int_XX
    )

In the same spirit, we specify control Hamiltonians to drive the system.
Again “connected” connected tells us which qubit this drive acts on and
“name” will later be used to assign the correct control signal to this
drive line.

.. code-block:: python

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

SPAM errors
^^^^^^^^^^^

In experimental practice, the qubit state can be mis-classified during
read-out. We simulate this by constructing a *confusion matrix*,
containing the probabilities for one qubit state being mistaken for
another.

.. code-block:: python

    m00_q1 = 0.97 # Prop to read qubit 1 state 0 as 0
    m01_q1 = 0.04 # Prop to read qubit 1 state 0 as 1
    m00_q2 = 0.96 # Prop to read qubit 2 state 0 as 0
    m01_q2 = 0.05 # Prop to read qubit 2 state 0 as 1
    one_zeros = np.array([0] * qubit_lvls)
    zero_ones = np.array([1] * qubit_lvls)
    one_zeros[0] = 1
    zero_ones[0] = 0
    val1 = one_zeros * m00_q1 + zero_ones * m01_q1
    val2 = one_zeros * m00_q2 + zero_ones * m01_q2
    min_val = one_zeros * 0.8 + zero_ones * 0.0
    max_val = one_zeros * 1.0 + zero_ones * 0.2
    confusion_row1 = Qty(value=val1, min_val=min_val, max_val=max_val, unit="")
    confusion_row2 = Qty(value=val2, min_val=min_val, max_val=max_val, unit="")
    conf_matrix = tasks.ConfusionMatrix(Q1=confusion_row1, Q2=confusion_row2)

The following task creates an initial thermal state with given
temperature.

.. code-block:: python

    init_temp = 50e-3
    init_ground = tasks.InitialiseGround(
        init_temp=Qty(
            value=init_temp,
            min_val=-0.001,
            max_val=0.22,
            unit='K'
        )
    )

We collect the parts specified above in the Model.

.. code-block:: python

    model = Mdl(
        [q1, q2], # Individual, self-contained components
        [drive, drive2, q1q2],  # Interactions between components
        [conf_matrix, init_ground] # SPAM processing
    )

Further, we can decide between coherent or open-system dynamics using
set_lindbladian() and whether to eliminate the static coupling by going
to the dressed frame with set_dressed().

.. code:: ipython3

    model.set_lindbladian(False)
    model.set_dressed(True)

Control signals
^^^^^^^^^^^^^^^

With the system model taken care of, we now specify the control
electronics and signal chain. Complex shaped controls are often realized
by creating an envelope signal with an arbitrary waveform generator
(AWG) with limited bandwith and mixing it with a fast, stable local
oscillator (LO).

.. code-block:: python

    sim_res = 100e9 # Resolution for numerical simulation
    awg_res = 2e9 # Realistic, limited resolution of an AWG
    lo = devices.LO(name='lo', resolution=sim_res)
    awg = devices.AWG(name='awg', resolution=awg_res)
    mixer = devices.Mixer(name='mixer')

Waveform generators exhibit a rise time, the time it takes until the
target voltage is set. This has a smoothing effect on the resulting
pulse shape.

.. code-block:: python

    resp = devices.Response(
        name='resp',
        rise_time=Qty(
            value=0.3e-9,
            min_val=0.05e-9,
            max_val=0.6e-9,
            unit='s'
        ),
        resolution=sim_res
    )

In simulation, we translate between AWG resolution and simulation (or
“analog”) resolution by including an up-sampling device.

.. code-block:: python

    dig_to_an = devices.DigitalToAnalog(
        name="dac",
        resolution=sim_res
    )

Control electronics apply voltages to lines, whereas in a Hamiltonian we
usually write the control fields in energy or frequency units. In
practice, this conversion can be highly non-trivial if it involves
multiple stages of attenuation and for example the conversion of a line
voltage in an antenna to a dipole field coupling to the qubit. The
following device represents a simple, linear conversion factor.

.. code-block:: python

    v2hz = 1e9
    v_to_hz = devices.VoltsToHertz(
        name='v_to_hz',
        V_to_Hz=Qty(
            value=v2hz,
            min_val=0.9e9,
            max_val=1.1e9,
            unit='Hz/V'
        )
    )

The generator combines the parts of the signal generation and assigns a
signal chain to each control line.

.. code-block:: python

    generator = Gnr(
            devices={
                "LO": devices.LO(name='lo', resolution=sim_res, outputs=1),
                "AWG": devices.AWG(name='awg', resolution=awg_res, outputs=1),
                "DigitalToAnalog": devices.DigitalToAnalog(
                    name="dac",
                    resolution=sim_res,
                    inputs=1,
                    outputs=1
                ),
                "Response": devices.Response(
                    name='resp',
                    rise_time=Qty(
                        value=0.3e-9,
                        min_val=0.05e-9,
                        max_val=0.6e-9,
                        unit='s'
                    ),
                    resolution=sim_res,
                    inputs=1,
                    outputs=1
                ),
                "Mixer": devices.Mixer(name='mixer', inputs=2, outputs=1),
                "VoltsToHertz": devices.VoltsToHertz(
                    name='v_to_hz',
                    V_to_Hz=Qty(
                        value=1e9,
                        min_val=0.9e9,
                        max_val=1.1e9,
                        unit='Hz/V'
                    ),
                    inputs=1,
                    outputs=1
                )
            },
            chains= {
                "d1": ["LO", "AWG", "DigitalToAnalog", "Response", "Mixer", "VoltsToHertz"],
                "d2": ["LO", "AWG", "DigitalToAnalog", "Response", "Mixer", "VoltsToHertz"]
            }
        )

Gates-set and Parameter map
^^^^^^^^^^^^^^^^^^^^^^^^^^^

It remains to write down what kind of operations we want to perform on
the device. For a gate based quantum computing chip, we define a
gate-set.

We choose a gate time of 7ns and a Gaussian envelope shape with a list
of parameters.

.. code-block:: python

    t_final = 7e-9   # Time for single qubit gates
    sideband = 50e6
    gauss_params_single = {
        'amp': Qty(
            value=0.5,
            min_val=0.4,
            max_val=0.6,
            unit="V"
        ),
        't_final': Qty(
            value=t_final,
            min_val=0.5 * t_final,
            max_val=1.5 * t_final,
            unit="s"
        ),
        'sigma': Qty(
            value=t_final / 4,
            min_val=t_final / 8,
            max_val=t_final / 2,
            unit="s"
        ),
        'xy_angle': Qty(
            value=0.0,
            min_val=-0.5 * np.pi,
            max_val=2.5 * np.pi,
            unit='rad'
        ),
        'freq_offset': Qty(
            value=-sideband - 3e6 ,
            min_val=-56 * 1e6 ,
            max_val=-52 * 1e6 ,
            unit='Hz 2pi'
        ),
        'delta': Qty(
            value=-1,
            min_val=-5,
            max_val=3,
            unit=""
        )
    }

Here we take ``gaussian_nonorm()`` from the libraries as the function to
define the shape.

.. code-block:: python

    gauss_env_single = pulse.Envelope(
        name="gauss",
        desc="Gaussian comp for single-qubit gates",
        params=gauss_params_single,
        shape=envelopes.gaussian_nonorm
    )

We also define a gate that represents no driving.

.. code-block:: python

    nodrive_env = pulse.Envelope(
        name="no_drive",
        params={
            't_final': Qty(
                value=t_final,
                min_val=0.5 * t_final,
                max_val=1.5 * t_final,
                unit="s"
            )
        },
        shape=envelopes.no_drive
    )

We specify the drive tones with an offset from the qubit frequencies. As
is done in experiment, we will later adjust the resonance by modulating
the envelope function.

.. code-block:: python

    lo_freq_q1 = 5e9  + sideband
    carrier_parameters = {
        'freq': Qty(
            value=lo_freq_q1,
            min_val=4.5e9 ,
            max_val=6e9 ,
            unit='Hz 2pi'
        ),
        'framechange': Qty(
            value=0.0,
            min_val= -np.pi,
            max_val= 3 * np.pi,
            unit='rad'
        )
    }
    carr = pulse.Carrier(
        name="carrier",
        desc="Frequency of the local oscillator",
        params=carrier_parameters
    )

For the second qubit drive tone, we copy the first one and replace the
frequency. The deepcopy is to ensure that we don’t just create a pointer
to the first drive.

.. code-block:: python

    lo_freq_q2 = 5.6e9  + sideband
    carr_2 = copy.deepcopy(carr)
    carr_2.params['freq'].set_value(lo_freq_q2)

Instructions
^^^^^^^^^^^^

We define the gates we want to perform with a “name” that will identify
them later and “channels” relating to the control Hamiltonians and drive
lines we specified earlier. As a start we write down 90 degree rotations
in the positive :math:`x`-direction and identity gates for both qubits.
Then we add a carrier and envelope to each.

.. code-block:: python

    rx90p_q1 = gates.Instruction(
        name="rx90p", targets=[0], t_start=0.0, t_end=t_final, channels=["d1", "d2"]
    )
    rx90p_q2 = gates.Instruction(
        name="rx90p", targets=[1], t_start=0.0, t_end=t_final, channels=["d1", "d2"]
    )

    rx90p_q1.add_component(gauss_env_single, "d1")
    rx90p_q1.add_component(carr, "d1")


    rx90p_q2.add_component(copy.deepcopy(gauss_env_single), "d2")
    rx90p_q2.add_component(carr_2, "d2")


When later compiling gates into sequences, we have to take care of the
relative rotating frames of the qubits and local oscillators. We do this
by adding a phase after each gate that realigns the frames.

.. code-block:: python

    rx90p_q1.add_component(nodrive_env, "d2")
    rx90p_q1.add_component(copy.deepcopy(carr_2), "d2")
    rx90p_q1.comps["d2"]["carrier"].params["framechange"].set_value(
        (-sideband * t_final) * 2 * np.pi % (2 * np.pi)
    )

    rx90p_q2.add_component(nodrive_env, "d1")
    rx90p_q2.add_component(copy.deepcopy(carr), "d1")
    rx90p_q2.comps["d1"]["carrier"].params["framechange"].set_value(
        (-sideband * t_final) * 2 * np.pi % (2 * np.pi)
    )

The remainder of the gates-set can be derived from the RX90p gate by
shifting its phase by multiples of :math:`\pi/2`.

.. code-block:: python

    ry90p_q1 = copy.deepcopy(rx90p_q1)
    ry90p_q1.name = "ry90p"
    rx90m_q1 = copy.deepcopy(rx90p_q1)
    rx90m_q1.name = "rx90m"
    ry90m_q1 = copy.deepcopy(rx90p_q1)
    ry90m_q1.name = "ry90m"
    ry90p_q1.comps['d1']['gauss'].params['xy_angle'].set_value(0.5 * np.pi)
    rx90m_q1.comps['d1']['gauss'].params['xy_angle'].set_value(np.pi)
    ry90m_q1.comps['d1']['gauss'].params['xy_angle'].set_value(1.5 * np.pi)
    single_q_gates = [rx90p_q1, ry90p_q1, rx90m_q1, ry90m_q1]


    ry90p_q2 = copy.deepcopy(rx90p_q2)
    ry90p_q2.name = "ry90p"
    rx90m_q2 = copy.deepcopy(rx90p_q2)
    rx90m_q2.name = "rx90m"
    ry90m_q2 = copy.deepcopy(rx90p_q2)
    ry90m_q2.name = "ry90m"
    ry90p_q2.comps['d2']['gauss'].params['xy_angle'].set_value(0.5 * np.pi)
    rx90m_q2.comps['d2']['gauss'].params['xy_angle'].set_value(np.pi)
    ry90m_q2.comps['d2']['gauss'].params['xy_angle'].set_value(1.5 * np.pi)
    single_q_gates.extend([rx90p_q2, ry90p_q2, rx90m_q2, ry90m_q2])

With every component defined, we collect them in the parameter map, our
object that holds information and methods to manipulate and examine
model and control parameters.

.. code-block:: python

    parameter_map = PMap(instructions=all_1q_gates_comb, model=model, generator=generator)

The experiment
^^^^^^^^^^^^^^

Finally everything is collected in the experiment that provides the
functions to interact with the system.

.. code-block:: python

    exp = Exp(pmap=parameter_map)

Simulation
^^^^^^^^^^

With our experiment all set-up, we can perform simulations. We first
decide which basic gates to simulate, in this case only the 90 degree
rotation on one qubit and the identity.

.. code-block:: python

    exp.set_opt_gates(['RX90p:Id', 'Id:Id'])

The actual numerical simulation is done by calling ``exp.compute_propagators()``.
This is the most resource intensive part as it involves solving the
equations of motion for the system.

.. code-block:: python

    unitaries = exp.compute_propagators()


.. code-block:: python



Dynamics
~~~~~~~~

To investigate dynamics, we define the ground state as an initial state.

.. code:: ipython3

    psi_init = [[0] * 9]
    psi_init[0][0] = 1
    init_state = tf.transpose(tf.constant(psi_init, tf.complex128))

.. code:: ipython3

    init_state




.. parsed-literal::

    <tf.Tensor: shape=(9, 1), dtype=complex128, numpy=
    array([[1.+0.j],
           [0.+0.j],
           [0.+0.j],
           [0.+0.j],
           [0.+0.j],
           [0.+0.j],
           [0.+0.j],
           [0.+0.j],
           [0.+0.j]])>



Since we stored the process matrices, we can now relatively inexpensively
evaluate sequences. We start with just one gate

.. code:: ipython3

    barely_a_seq = ['rx90p[0]']

and plot system dynamics.

.. code-block:: python

    def plot_dynamics(exp, psi_init, seq, goal=-1):
            """
            Plotting code for time-resolved populations.

            Parameters
            ----------
            psi_init: tf.Tensor
                Initial state or density matrix.
            seq: list
                List of operations to apply to the initial state.
            goal: tf.float64
                Value of the goal function, if used.
            debug: boolean
                If true, return a matplotlib figure instead of saving.
            """
            model = exp.pmap.model
            dUs = exp.partial_propagators
            psi_t = psi_init.numpy()
            pop_t = exp.populations(psi_t, model.lindbladian)
            for gate in seq:
                for du in dUs[gate]:
                    psi_t = np.matmul(du.numpy(), psi_t)
                    pops = exp.populations(psi_t, model.lindbladian)
                    pop_t = np.append(pop_t, pops, axis=1)

            fig, axs = plt.subplots(1, 1)
            ts = exp.ts
            dt = ts[1] - ts[0]
            ts = np.linspace(0.0, dt*pop_t.shape[1], pop_t.shape[1])
            axs.plot(ts / 1e-9, pop_t.T)
            axs.grid(linestyle="--")
            axs.tick_params(
                direction="in", left=True, right=True, top=True, bottom=True
            )
            axs.set_xlabel('Time [ns]')
            axs.set_ylabel('Population')
            plt.legend(model.state_labels)
            pass



.. code-block:: python

    plot_dynamics(exp, init_state, barely_a_seq)



.. image:: dyn_singleX.png


We can see an ill-defined un-optimized gate. The labels indicate qubit
states in the product basis. Next we increase the number of repetitions
of the same gate.

.. code-block:: python

    barely_a_seq * 10




.. parsed-literal::

    ['rx90p[0]',
     'rx90p[0]',
     'rx90p[0]',
     'rx90p[0]',
     'rx90p[0]',
     'rx90p[0]',
     'rx90p[0]',
     'rx90p[0]',
     'rx90p[0]',
     'rx90p[0]']



.. code-block:: python

    plot_dynamics(exp, init_state, barely_a_seq * 5)



.. image:: dyn_5X.png


.. code-block:: python

    plot_dynamics(exp, init_state, barely_a_seq * 10)



.. image:: dyn_10X.png


Note that at this point, we only multiply already computed matrices. We
don’t need to solve the equations of motion again for new sequences.
