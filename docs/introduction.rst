Introduction
=============

Efforts to scale-up quantum computation have reached a point where the principal limiting factor is
not the number of qubits, but the entangling gate infidelity.
However, highly detailed system characterization is an arduous process, thus the underlying errors are
rarely well understood.
Open-loop optimal control techniques allow for the improvement of gates but are limited by the models 
they are based on.

To rectify the situation, we provide a new integrated tool-set for Control, Calibration and Characterization, 
comprised of open-loop pulse optimization, model-free calibration, model fitting and refinement.
We present a methodology to combine these tools to find a quantitatively accurate system model,
high-fidelity gates and an approximate error budget, all based on a high-performance, feature-rich 
simulator.
We present a working example of the method on a superconducting 2-qubit QPU. 
Starting from a roughly  characterized system, we are able to learn the model parameters to high accuracy and 
derive a coherence limited 99.6\% CR gate.

C3 software is made available under the Apache 2.0 open-source license at www.q-optimize.org