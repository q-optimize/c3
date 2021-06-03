Introduction to :math:`C^3` Toolset
====================================

In this section, we go over the foundational components and concepts in :math:`C^3` with the 
primary objective of understanding how the different sub-modules inside the :code:`c3-toolset`
are structured, the purpose they serve and how to tie them together into a complete Automated
Quantum Device Bring-up workflow. For more detailed examples of how to use the :code:`c3-toolset`
to perform a specific Quantum Control task, please check out the :doc:`two_qubits` or the 
:doc:`Simulated_calibration` sections or refer to the :doc:`c3` for descriptions of 
Classes and Functions.


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

Classical Control Electronics
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
A digital twin of the electronic control stack associated with the Quantum Processing Unit. The
:code:`Generator` class contains the required encapsulation in the form of :code:`devices` which
help model the behaviour of the classical control electronics taking account of their imperfections and 
physical realisations. The devices e.g, an LO or an AWG or a Mixer are wired together in the 
:code:`Generator` object to form a complete representation of accessory electronics.

Instructions
~~~~~~~~~~~~~~
Once there is a software model for the QPU and the control electronics, one would need to define 
Instructions or operations to be perform on this device. For gate-based quantum computing , this is 
in the form of gates and their underlying pulse operations. Pulse shapes are described through a 
:code:`Envelope` along with a :code:`Carrier`, which are then wrapped up in the form of :code:`Instruction` 
objects. The sequence in which these gates are applied are not defined at this stage.


.. warning::
    Components inside the :code:`c3/generator/` and :code:`c3/signal/` sub-modules will be restructured 
    in an upcoming release to be more consistent with how the :code:`Model` class encapsulates smaller 
    blocks present in the :code:`c3/libraries` sub-module.


Parameter Map
--------------

The :code:`ParameterMap` helps to obtain an optimizable vector of parameters from the various theoretical 
models previously defined. This allows for a simple interface to the optimization algorithms which are tasked
with optimizing different sets of variables used to define some entity, e.g, optimizing pulse parameters by 
calibrating on hardware or providing an optimal gate-set through model-based quantum control.

Experiments
-------------

With the building blocks in place, we can bring them all together through an :code:`Experiment` object that
encapsulates the device model, the control signals, the instructions and the parameter map. Note that depending on
the use only some of the blocks are essential when building the experiment.

Optimizers
-----------

At its core, :code:`c3-toolset` is an optimization framework and all of the three steps - Open-Loop, Calibration and 
Model Learning can be defined as a optimization task. The :code:`optimizers` contain classes that provide 
helpful encapsulation for these steps. These objects take as arguments the previously defined :code:`Experiment` and 
:code:`ParameterMap` objects along with an :code:`algorithm` e.g, :code:`CMA-eS` or :code:`L-BFGS` which performs 
the iterative optimization steps.

Libraries
----------

The :code:`c3/libraries` sub-module includes various helpful library of components that are used somewhat like lego
pieces when building the bigger blocks, e.g, :code:`hamiltonians` for the :code:`chip` present in the :code:`Model`
or :code:`envelopes` defining a control :code:`pulse`. More details about these components are available in the
:doc:`c3.libraries` section.
