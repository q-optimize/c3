Logs and current optimization status
====================================

During optimizations (optimal control, calibration, model learning), a
current best point is stored in the log folder to monitor progress.
Called on a log file it will print a
`rich <https://github.com/willmcgugan/rich>`__ table of the current
status. With the ``-w`` or ``-- watch`` options the table will keep
updating.

.. code:: bash

    c3/utils/log_reader.py -h


.. code-block::

    usage: log_reader.py [-h] [-w WATCH] log_file
    
    positional arguments:
      log_file
    
    optional arguments:
      -h, --help            show this help message and exit
      -w WATCH, --watch WATCH
                            Update the table every WATCH seconds.


Using the example log from the test folder:

.. code:: bash

    c3/utils/log_reader.py test/sample_optim_log.c3log


.. parsed-literal::

    Optimization reached 0.00462 at Tue Aug 17 15:28:09 2021
    
    ┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━┓
    ┃ Parameter                     ┃ Value           ┃ Gradient         ┃
    ┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━┩
    │ rx90p[0]-d1-gauss-amp         │      497.311 mV │        18.720 mV │
    │ rx90p[0]-d1-gauss-freq_offset │ -52.998 MHz 2pi │ -414.237 µHz 2pi │
    │ rx90p[0]-d1-gauss-xy_angle    │    -47.409 mrad │       2.904 mrad │
    │ rx90p[0]-d1-gauss-delta       │          -1.077 │          6.648 m │
    └───────────────────────────────┴─────────────────┴──────────────────┘

