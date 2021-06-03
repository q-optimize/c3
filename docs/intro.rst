Introduction to :math:`C^3` Toolset
====================================

In this section, we go over the foundational components and concepts in :math:`C^3` with the 
primary objective of understanding how the different sub-modules inside the :code:`c3-toolset`
are structured, the purpose they serve and how to tie them together into a complete Automated
Quantum Device Bring-up workflow. For more detailed examples of how to use the :code:`c3-toolset`
to perform a specific Quantum Control task, please check out the :doc:`two_qubits` or the 
:doc:`Simulated_calibration` sections


The Building Blocks
--------------------

There are three basic building blocks that form the foundation of all the modelling and calibration 
tasks one can perform using :code:`c3-toolset`, and depending on the use-case, some or all of these
blocks might be useful. These are the following:

- Quantum Device Model 
- Classical Control Electronics
- Instructions

Quantum Device Model
~~~~~~~~~~~~~~~~~~~~~

A theoretical Physics-based model of the Quantum Processing Unit. This is encapsulated by the 
:code:`Model` class which consists of objects from the :code:`chip` and :code:`tasks` library.
:code:`chip` contains Hamiltonian models of different kinds of qubit realisations, along with
their couplings while :code:`tasks` let you perform common operations such as qubit initialisation or
readout. A typical :code:`Model` object would contain objects encapsulating qubits along with their 
interactions as drive lines and tasks, if any.
