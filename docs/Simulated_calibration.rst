Simulated calibration with :math:`C_2`
======================================

Calibration of control pulses is the process of fine-tuning parameters
in a feedback-loop with the experiment. We will simulate this process
here by constructing a black-box simulation and interacting with it
exactly like an experiment.

We have manange imports and creation of the black-box the same way as in
the previous example in a helper ``single_qubit_blackbox_exp.py``.

.. code-block:: python

    from single_qubit_blackbox_exp import create_experiment

    blackbox = create_experiment()

This blackbox is constructed the same way as in the OptimalControl example. The
difference will be in how we interact with it. First, we decide on what
experiment we want to perform and need to specify it as a python
function. A general, minimal example would be

.. code-block:: python

    def exp_communication(params):
      # Send parameters to experiment controller
      # and receive a measurement result.
      return measurement_result


Again, ``params`` is a linear vector of bare numbers. The measurement
result can be a single number or a set of results. It can also include
additional information about statistics, like averaging, standard
deviation, etc.

ORBIT - Single-length randomized benchmarking
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The following defines an `ORBIT <https://arxiv.org/abs/1403.0035>`__
procedure. In short, we define sequences of gates that result in an
identity gate if our individual gates are perfect. Any deviation from
identity gives us a measure of the imperfections in our gates. Our
helper ``qt_utils`` provides these sequences.

.. code-block:: python

    from c3.utils import qt_utils

.. code-block:: python

    qt_utils.single_length_RB(
                RB_number=1, RB_length=5, target=0
        )




.. parsed-literal::

    [['ry90m[0]',
      'rx90p[0]',
      'rx90m[0]',
      'rx90p[0]',
      'ry90p[0]',
      'ry90p[0]',
      'ry90p[0]',
      'rx90p[0]',
      'ry90m[0]',
      'rx90p[0]']]



The desired number of 5 gates is selected from a specific set (the
Clifford group) and has to be decomposed into the available gate-set.
Here, this means 4 gates per Clifford, hence a sequence of 20 gates.

Communication with the experiment
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Some of the following code is specific to the fact that this a
*simulated* calibration. The interface of :math:`C^2` to the experiment
is simple: parameters in :math:`\rightarrow` results out. Thus, we have
to wrap the blackbox by defining the target states and the ``opt_map``.

.. code-block:: python

    import numpy as np
    import tensorflow as tf

    def ORBIT_wrapper(p):
        def ORBIT(params, exp, opt_map, qubit_labels, logdir):
            ### ORBIT meta-parameters ###
            RB_length = 60 # How long each sequence is
            RB_number = 40  # How many sequences
            shots = 1000    # How many averages per readout

            ################################
            ### Simulation specific part ###
            ################################

            do_noise = False  # Whether to add artificial noise to the results

            qubit_label = list(qubit_labels.keys())[0]
            state_labels = qubit_labels[qubit_label]
            state_label = [tuple(l) for l in state_labels]

            # Creating the RB sequences #
            seqs = qt_utils.single_length_RB(
                    RB_number=RB_number, RB_length=RB_length, target=0
            )

            # Transmitting the parameters to the experiment #
            exp.pmap.set_parameters(params, opt_map)
            exp.set_opt_gates_seq(seqs)

            # Simulating the gates #
            U_dict = exp.compute_propagators()

            # Running the RB sequences and read-out the results #
            pops = exp.evaluate(seqs)
            pop1s, _ = exp.process(pops, labels=state_label)

            results = []
            results_std = []
            shots_nums = []

            # Collecting results and statistics, add noise #
            if do_noise:
                for p1 in pop1s:
                    draws = tf.keras.backend.random_binomial(
                        [shots],
                        p=p1[0],
                        dtype=tf.float64,
                    )
                    results.append([np.mean(draws)])
                    results_std.append([np.std(draws)/np.sqrt(shots)])
                    shots_nums.append([shots])
            else:
                for p1 in pop1s:
                    results.append(p1.numpy())
                    results_std.append([0])
                    shots_nums.append([shots])

            #######################################
            ### End of Simulation specific part ###
            #######################################

            goal = np.mean(results)
            return goal, results, results_std, seqs, shots_nums
        return ORBIT(
                    p, blackbox, gateset_opt_map, state_labels, "/tmp/c3logs/blackbox"
                )

Optimization
~~~~~~~~~~~~

We first import algorithms and the correct optimizer object.

.. code-block:: python

    import copy

    from c3.experiment import Experiment as Exp
    from c3.c3objs import Quantity as Qty
    from c3.parametermap import ParameterMap as PMap
    from c3.libraries import algorithms, envelopes
    from c3.signal import gates, pulse
    from c3.optimizers.calibration import Calibration

Representation of the experiment within :math:`C^3`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

At this point we have to make sure that the gates (“RX90p”, etc.) and
drive line (“d1”) are compatible to the experiment controller operating
the blackbox. We mirror the blackbox by creating an experiment in the
:math:`C^3` context:

.. code-block:: python

    t_final = 7e-9   # Time for single qubit gates
    sideband = 50e6
    lo_freq = 5e9 + sideband

     # ### MAKE GATESET
    gauss_params_single = {
        'amp': Qty(
            value=0.45,
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
            value=-sideband - 0.5e6,
            min_val=-53 * 1e6,
            max_val=-47 * 1e6,
            unit='Hz 2pi'
        ),
        'delta': Qty(
            value=-1,
            min_val=-5,
            max_val=3,
            unit=""
        )
    }

    gauss_env_single = pulse.Envelope(
        name="gauss",
        desc="Gaussian comp for single-qubit gates",
        params=gauss_params_single,
        shape=envelopes.gaussian_nonorm
    )
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
    carrier_parameters = {
        'freq': Qty(
            value=lo_freq,
            min_val=4.5e9,
            max_val=6e9,
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

    rx90p = gates.Instruction(
        name="rx90p",
        t_start=0.0,
        t_end=t_final,
        channels=["d1"]
    )
    QId = gates.Instruction(
        name="id",
        t_start=0.0,
        t_end=t_final,
        channels=["d1"]
    )

    rx90p.add_component(gauss_env_single, "d1")
    rx90p.add_component(carr, "d1")
    QId.add_component(nodrive_env, "d1")
    QId.add_component(copy.deepcopy(carr), "d1")
    QId.comps['d1']['carrier'].params['framechange'].set_value(
        (-sideband * t_final * 2 * np.pi) % (2*np.pi)
    )
    ry90p = copy.deepcopy(rx90p)
    ry90p.name = "ry90p"
    rx90m = copy.deepcopy(rx90p)
    rx90m.name = "rx90m"
    ry90m = copy.deepcopy(rx90p)
    ry90m.name = "ry90m"
    ry90p.comps['d1']['gauss'].params['xy_angle'].set_value(0.5 * np.pi)
    rx90m.comps['d1']['gauss'].params['xy_angle'].set_value(np.pi)
    ry90m.comps['d1']['gauss'].params['xy_angle'].set_value(1.5 * np.pi)

    parameter_map = PMap(instructions=[QId, rx90p, ry90p, rx90m, ry90m])

    # ### MAKE EXPERIMENT
    exp = Exp(pmap=parameter_map)

Next, we define the parameters we whish to calibrate. See how these gate
instructions are defined in the experiment setup example or in
``single_qubit_blackbox_exp.py``. Our gate-set is made up of 4 gates,
rotations of 90 degrees around the :math:`x` and :math:`y`-axis in
positive and negative direction. While it is possible to optimize each
parameters of each gate individually, in this example all four gates
share parameters. They only differ in the phase :math:`\phi_{xy}` that
is set in the definitions.

.. code-block:: python

    gateset_opt_map =   [
        [
          ("rx90p[0]", "d1", "gauss", "amp"),
          ("ry90p[0]", "d1", "gauss", "amp"),
          ("rx90m[0]", "d1", "gauss", "amp"),
          ("ry90m[0]", "d1", "gauss", "amp")
        ],
        [
          ("rx90p[0]", "d1", "gauss", "delta"),
          ("ry90p[0]", "d1", "gauss", "delta"),
          ("rx90m[0]", "d1", "gauss", "delta"),
          ("ry90m[0]", "d1", "gauss", "delta")
        ],
        [
          ("rx90p[0]", "d1", "gauss", "freq_offset"),
          ("ry90p[0]", "d1", "gauss", "freq_offset"),
          ("rx90m[0]", "d1", "gauss", "freq_offset"),
          ("ry90m[0]", "d1", "gauss", "freq_offset")
        ],
        [
          ("id[0]", "d1", "carrier", "framechange")
        ]
      ]

    parameter_map.set_opt_map(gateset_opt_map)

As defined above, we have 16 parameters where 4 share their numerical
value. This leaves 4 values to optimize.

.. code-block:: python

    parameter_map.print_parameters()


.. parsed-literal::

    rx90p[0]-d1-gauss-amp                 : 450.000 mV
    ry90p[0]-d1-gauss-amp
    rx90m[0]-d1-gauss-amp
    ry90m[0]-d1-gauss-amp

    rx90p[0]-d1-gauss-delta               : -1.000
    ry90p[0]-d1-gauss-delta
    rx90m[0]-d1-gauss-delta
    ry90m[0]-d1-gauss-delta

    rx90p[0]-d1-gauss-freq_offset         : -50.500 MHz 2pi
    ry90p[0]-d1-gauss-freq_offset
    rx90m[0]-d1-gauss-freq_offset
    ry90m[0]-d1-gauss-freq_offset

    id[0]-d1-carrier-framechange          : 4.084 rad


It is important to note that in this example, we are transmitting only
these four parameters to the experiment. We don’t know how the blackbox
will implement the pulse shapes and care has to be taken that the
parameters are understood on the other end. Optionally, we could
specifiy a virtual AWG within :math:`C^3` and transmit pixilated pulse
shapes directly to the physiscal AWG.

Algorithms
~~~~~~~~~~

As an optimization algoritm, we choose
`CMA-Es <https://en.wikipedia.org/wiki/CMA-ES>`__ and set up some
options specific to this algorithm.

.. code-block:: python

    alg_options = {
        "popsize" : 10,
        "maxfevals" : 300,
        "init_point" : "True",
        "tolfun" : 0.01,
        "spread" : 0.25
      }

We define the subspace as both excited states :math:`\{|1>,|2>\}`,
assuming read-out can distinguish between 0, 1 and 2.

.. code-block:: python

    state_labels = {
          "excited" : [(1,), (2,)]
      }

In the real world, this setup needs to be handled in the experiment
controller side. We construct the optimizer object with the options we
setup:

.. code-block:: python

    import os
    import tempfile

    # Create a temporary directory to store logfiles, modify as needed
    log_dir = os.path.join(tempfile.TemporaryDirectory().name, "c3logs")

    opt = Calibration(
        dir_path=log_dir,
        run_name="ORBIT_cal",
        eval_func=ORBIT_wrapper,
        pmap=parameter_map,
        exp_right=exp,
        algorithm=algorithms.cmaes,
        options=alg_options
    )
    opt.set_exp(exp)

And run the calibration:

.. code-block:: python

    x = parameter_map.get_parameters_scaled()

.. code:: ipython3

    opt.optimize_controls()



    C3:STATUS:Saving as: /tmp/tmpicnnbliz/c3logs/ORBIT_cal/2021_01_28_T_15_17_30/calibration.log
    (5_w,10)-aCMA-ES (mu_w=3.2,w_1=45%) in dimension 4 (seed=912463, Thu Jan 28 15:17:30 2021)
    C3:STATUS:Adding initial point to CMA sample.
    Iterat #Fevals   function value  axis ratio  sigma  min&max std  t[m:s]
        1     10 1.446744168975211e-01 1.0e+00 2.11e-01  2e-01  2e-01 1:18.9
        2     20 2.074359374665050e-01 1.4e+00 1.96e-01  1e-01  2e-01 2:28.5
        3     30 1.042216610303495e-01 1.5e+00 1.76e-01  1e-01  2e-01 3:36.4
        4     40 1.720244494886762e-01 1.9e+00 1.88e-01  1e-01  2e-01 4:46.5
        5     50 9.761264536669531e-02 2.2e+00 2.05e-01  1e-01  2e-01 6:15.4
        6     60 1.956493007802809e-01 2.8e+00 1.75e-01  8e-02  2e-01 7:17.9
        7     70 6.625917264980545e-02 3.0e+00 2.20e-01  9e-02  3e-01 8:22.8
        8     80 7.697621753428294e-02 4.1e+00 2.19e-01  8e-02  3e-01 9:25.8
        9     90 8.826758030850271e-02 4.7e+00 1.85e-01  6e-02  3e-01 10:28.7
       10    100 9.099567192014653e-02 5.3e+00 1.59e-01  4e-02  2e-01 11:32.7
       11    110 6.673347151005890e-02 6.9e+00 1.49e-01  3e-02  2e-01 12:27.9
       12    120 6.822093884865452e-02 7.6e+00 1.68e-01  4e-02  2e-01 13:26.6
       13    130 6.307315835232992e-02 8.1e+00 1.42e-01  3e-02  2e-01 14:22.8
       14    140 6.301017013241370e-02 7.8e+00 1.42e-01  2e-02  2e-01 15:18.7
       15    150 6.795728963072037e-02 9.3e+00 1.32e-01  2e-02  2e-01 16:15.8
       16    160 7.675314380135559e-02 9.2e+00 1.03e-01  2e-02  1e-01 17:12.9
       17    170 6.806172046778505e-02 9.1e+00 8.05e-02  1e-02  1e-01 18:11.5
       18    180 5.698438523961635e-02 1.0e+01 7.42e-02  9e-03  9e-02 19:06.1
       19    190 5.536707419037251e-02 1.1e+01 6.89e-02  8e-03  9e-02 20:00.6
       20    200 4.924177790655197e-02 1.2e+01 7.31e-02  8e-03  9e-02 20:58.2
       21    210 5.836136870997249e-02 1.2e+01 8.20e-02  8e-03  1e-01 21:55.1
       22    220 5.463139088536284e-02 1.3e+01 8.29e-02  9e-03  1e-01 22:51.0
       23    230 4.562693294212217e-02 1.4e+01 8.66e-02  9e-03  1e-01 23:48.3
       24    240 5.188441161313757e-02 1.6e+01 7.74e-02  7e-03  1e-01 24:46.1
       25    250 5.199237655967553e-02 1.7e+01 7.41e-02  6e-03  9e-02 25:47.1
       26    260 5.684400595430246e-02 1.6e+01 6.41e-02  5e-03  9e-02 26:43.7
       27    270 4.441763519087279e-02 1.8e+01 5.12e-02  4e-03  7e-02 27:36.2
       28    280 4.994977609185950e-02 1.8e+01 5.51e-02  5e-03  8e-02 28:33.9
       29    290 6.108777009078262e-02 1.8e+01 5.14e-02  4e-03  7e-02 29:30.4
       30    300 5.658962789881571e-02 1.8e+01 4.65e-02  4e-03  6e-02 30:28.0
       31    310 5.765354335022381e-02 1.8e+01 4.77e-02  4e-03  6e-02 31:26.9
    termination on maxfevals=300
    final/bestever f-value = 5.765354e-02 4.441764e-02
    incumbent solution: [-0.4739081748676816, -0.09828275146514219, -1.0504851431889897, 0.9108808620989909]
    std deviation: [0.013780217516583012, 0.0038070906112681576, 0.02460767003734409, 0.05816700836608336]


Analysis
~~~~~~~~

The following code uses matplotlib to create an ORBIT plot from the
logfile.

.. code-block:: python

    import json
    from matplotlib.ticker import MaxNLocator
    from  matplotlib import rcParams
    from matplotlib import cycler
    import matplotlib as mpl
    import matplotlib.pyplot as plt

    rcParams['xtick.direction'] = 'in'
    rcParams['axes.grid'] = True
    rcParams['grid.linestyle'] = '--'
    rcParams['markers.fillstyle'] = 'none'
    rcParams['axes.prop_cycle'] = cycler(
        'linestyle', ["-", "--"]
    )
    rcParams['text.usetex'] = True
    rcParams['font.size'] = 16
    rcParams['font.family'] = 'serif'

    logfilename = opt.logdir + "calibration.log"
    with open(logfilename, "r") as filename:
        log = filename.readlines()


    options = json.loads(log[7])

    goal_function = []
    batch = 0
    batch_size = options["popsize"]


    eval = 0
    for line in log[9:]:
        if line[0] == "{":
            if not eval % batch_size:
                batch = eval // batch_size
                goal_function.append([])
            eval += 1
            point = json.loads(line)
            if 'goal' in point.keys():
                goal_function[batch].append(point['goal'])

    # Clean unfinished batch
    if len(goal_function[-1])<batch_size:
        goal_function.pop(-1)

    fig, ax = plt.subplots(1)
    means = []
    bests = []
    for ii in range(len(goal_function)):
        means.append(np.mean(np.array(goal_function[ii])))
        bests.append(np.min(np.array(goal_function[ii])))
        for pt in goal_function[ii]:
            ax.plot(ii+1, pt, color='tab:blue', marker="D", markersize=2.5, linewidth=0)

    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.set_ylabel('ORBIT')
    ax.set_xlabel('Iterations')
    ax.plot(
        range(1, len(goal_function)+1), bests, color="tab:red", marker="D",
        markersize=5.5, linewidth=0, fillstyle='full'
    )


.. image:: ORBIT.png
