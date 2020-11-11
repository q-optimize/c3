.. c3 documentation master file, created by
   sphinx-quickstart on Tue Aug 25 10:24:31 2020.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

===========================================================================

.. toctree::
   :maxdepth: 4
   :caption: API documentation:

===================================================================================
:math:`C^3` -  An integrated tool-set for control, calibration and characterization
===================================================================================

The :math:`C^3` software package provides tools to simulate and interact with experiments to perform common control and characterization tasks. Modules can be used individually or combined to achieve a certain goal. The main focus are three optimizations:

* :math:`C_1` Open-loop optimal control: Given a model, find the pulse shapes which maximize fidelity with a target operation.
* :math:`C_2`  Closed-loop calibration: Given pulses, calibrate their parameters to maximize a figure of merit measured by the actual experiment, thus improving beyond the limits of a deficient model.
* :math:`C_3`  Model learning: Given control pulses and their experimental measurement outcome, optimize model parameters to best reproduce the results.

When combined in sequence, these three procedures represent a recipe for system characterization.

*Note: This documentation is work-in-progress.*

.. toctree::
   :maxdepth: 4
   :caption: Contents:

   two_qubits
   optimal_control
   Simulated_calibration
   c3


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
