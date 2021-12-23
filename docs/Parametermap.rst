Parameter handling
==================

The tool within :math:`C^3` to manipulate the parameters of both the
model and controls is the ``ParameterMap``. It provides methods to
present the same data for human interaction, i.e. structured information
with physical units and for numerical optimization algorithms that
prefer a linear vector of scale 1. Here, we’ll show some example usage.
We’ll use the ``ParameterMap`` of the model also used in the simulated
calibration example.

.. code-block:: python

    from single_qubit_blackbox_exp import create_experiment
    
    exp = create_experiment()
    pmap = exp.pmap

The pmap contains a list of all parameters and their values:

.. code-block:: python

    pmap.get_full_params()




.. parsed-literal::

    {'Q1-freq': 5.000 GHz 2pi,
     'Q1-anhar': -210.000 MHz 2pi,
     'Q1-temp': 0.000 K,
     'init_ground-init_temp': -3.469 aK,
     'resp-rise_time': 300.000 ps,
     'v_to_hz-V_to_Hz': 1.000 GHz/V,
     'id[0]-d1-no_drive-amp': 1.000 V,
     'id[0]-d1-no_drive-delta': 0.000 V,
     'id[0]-d1-no_drive-freq_offset': 0.000 Hz 2pi,
     'id[0]-d1-no_drive-xy_angle': 0.000 rad,
     'id[0]-d1-no_drive-sigma': 5.000 ns,
     'id[0]-d1-no_drive-t_final': 7.000 ns,
     'id[0]-d1-carrier-freq': 5.050 GHz 2pi,
     'id[0]-d1-carrier-framechange': 5.933 rad,
     'rx90p[0]-d1-gauss-amp': 450.000 mV,
     'rx90p[0]-d1-gauss-delta': -1.000 ,
     'rx90p[0]-d1-gauss-freq_offset': -50.500 MHz 2pi,
     'rx90p[0]-d1-gauss-xy_angle': -444.089 arad,
     'rx90p[0]-d1-gauss-sigma': 1.750 ns,
     'rx90p[0]-d1-gauss-t_final': 7.000 ns,
     'rx90p[0]-d1-carrier-freq': 5.050 GHz 2pi,
     'rx90p[0]-d1-carrier-framechange': 0.000 rad,
     'ry90p[0]-d1-gauss-amp': 450.000 mV,
     'ry90p[0]-d1-gauss-delta': -1.000 ,
     'ry90p[0]-d1-gauss-freq_offset': -50.500 MHz 2pi,
     'ry90p[0]-d1-gauss-xy_angle': 1.571 rad,
     'ry90p[0]-d1-gauss-sigma': 1.750 ns,
     'ry90p[0]-d1-gauss-t_final': 7.000 ns,
     'ry90p[0]-d1-carrier-freq': 5.050 GHz 2pi,
     'ry90p[0]-d1-carrier-framechange': 0.000 rad,
     'rx90m[0]-d1-gauss-amp': 450.000 mV,
     'rx90m[0]-d1-gauss-delta': -1.000 ,
     'rx90m[0]-d1-gauss-freq_offset': -50.500 MHz 2pi,
     'rx90m[0]-d1-gauss-xy_angle': 3.142 rad,
     'rx90m[0]-d1-gauss-sigma': 1.750 ns,
     'rx90m[0]-d1-gauss-t_final': 7.000 ns,
     'rx90m[0]-d1-carrier-freq': 5.050 GHz 2pi,
     'rx90m[0]-d1-carrier-framechange': 0.000 rad,
     'ry90m[0]-d1-gauss-amp': 450.000 mV,
     'ry90m[0]-d1-gauss-delta': -1.000 ,
     'ry90m[0]-d1-gauss-freq_offset': -50.500 MHz 2pi,
     'ry90m[0]-d1-gauss-xy_angle': 4.712 rad,
     'ry90m[0]-d1-gauss-sigma': 1.750 ns,
     'ry90m[0]-d1-gauss-t_final': 7.000 ns,
     'ry90m[0]-d1-carrier-freq': 5.050 GHz 2pi,
     'ry90m[0]-d1-carrier-framechange': 0.000 rad}



To access a specific parameter, e.g. the frequency of qubit 1, we use
the identifying tuple ``('Q1','freq')``.

.. code-block:: python

    pmap.get_parameter(('Q1','freq'))




.. parsed-literal::

    5.000 GHz 2pi



The opt_map
-----------

To deal with multiple parameters we use the ``opt_map``, a nested list
of identifyers.

.. code-block:: python

    opt_map = [
        [
            ("Q1", "freq")
        ],
        [
            ("Q1", "anhar")
        ],  
    ]

Here, we get a list of the parameter values:

.. code-block:: python

    pmap.get_parameters(opt_map)




.. parsed-literal::

    [5.000 GHz 2pi, -210.000 MHz 2pi]



Let’s look at the amplitude values of two gaussian control pulses,
rotations about the :math:`X` and :math:`Y` axes repsectively.

.. code-block:: python

    opt_map = [
        [
            ('rx90p[0]','d1','gauss','amp')
        ],
        [
            ('ry90p[0]','d1','gauss','amp')
        ],  
    ]

.. code-block:: python

    pmap.get_parameters(opt_map)




.. parsed-literal::

    [450.000 mV, 450.000 mV]



We can set the parameters to new values.

.. code-block:: python

    pmap.set_parameters([0.5, 0.6], opt_map)

.. code-block:: python

    pmap.get_parameters(opt_map)




.. parsed-literal::

    [500.000 mV, 600.000 mV]



The opt_map also allows us to specify that two parameters should have
identical values. Here, let’s demand our :math:`X` and :math:`Y`
rotations use the same amplitude.

.. code-block:: python

    opt_map_ident = [
        [
            ('rx90p[0]','d1','gauss','amp'),
            ('ry90p[0]','d1','gauss','amp')
        ],
    ]

The grouping here means that these parameters share their numerical
value.

.. code-block:: python

    pmap.set_parameters([0.432], opt_map_ident)
    pmap.get_parameters(opt_map_ident)




.. parsed-literal::

    [432.000 mV]



.. code-block:: python

    pmap.get_parameters(opt_map)




.. parsed-literal::

    [432.000 mV, 432.000 mV]



During an optimization, the varied parameters do not change, so we fix
the opt_map

.. code-block:: python

    pmap.set_opt_map(opt_map)

.. code-block:: python

    pmap.get_parameters()




.. parsed-literal::

    [432.000 mV, 432.000 mV]



Optimizer scaling
-----------------

To be independent of the choice of numerical optimizer, they should use
the methods

.. code-block:: python

    pmap.get_parameters_scaled()




.. parsed-literal::

    array([-0.68, -0.68])



To provide values bound to :math:`[-1, 1]`. Let’s set the parameters to
their allowed minimum an maximum value with

.. code-block:: python

    pmap.set_parameters_scaled([1.0,-1.0])

.. code-block:: python

    pmap.get_parameters()




.. parsed-literal::

    [600.000 mV, 400.000 mV]



As a safeguard, when setting values outside of the unit range, their
physical values get looped back in the specified limits.

.. code-block:: python

    pmap.set_parameters_scaled([2.0, 3.0])

.. code-block:: python

    pmap.get_parameters()




.. parsed-literal::

    [500.000 mV, 400.000 mV]



Storing and reading
-------------------

For optimization purposes, we can store and load parameter values in
`HJSON <https://hjson.github.io/>`__ format.

.. code-block:: python

    pmap.store_values("current_vals.c3log")

.. code-block:: python

    !cat current_vals.c3log


.. parsed-literal::

    {
      opt_map:
      [
        [
          rx90p[0]-d1-gauss-amp
        ]
        [
          ry90p[0]-d1-gauss-amp
        ]
      ]
      units:
      [
        V
        V
      ]
      optim_status:
      {
        params:
        [
          0.5
          0.4000000059604645
        ]
      }
    }


.. code-block:: python

    pmap.load_values("current_vals.c3log")
