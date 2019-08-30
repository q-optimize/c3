"""C3PO configuration file"""

from numpy import pi

import numpy as np
import qutip as qt
import matplotlib.pyplot as plt

from qutip import *
import c3po

from c3po.cobj.component import *
from c3po.main.model import Model as mdl

import time

from tensorflow.python import debug as tf_debug

# to change to this once we want to do model learning
# def duffing(a, values):
#     c1 = lambda freq: freq
#     c2 = lambda delta: delta/2
#     cs = [c1(values['freq']),c2(values['delta'])]
#     H1 = a.dag() * a
#     H2 = (a.dag() * a - 1) * a.dag() * a
#     Hs = [H1,H2]
#     return Hs, cs

def duffing(a, values):
    return values['freq'] * a.dag() * a + values['delta']/2 * (a.dag() * a - 1) * a.dag() * a

def resonator(a, values):
    return values['freq'] * a.dag() * a

def int_XX(anhs, values):
    return values['strength'] * (anhs[0].dag() + anhs[0]) * (anhs[1].dag() + anhs[1])

def drive(anhs, values):
    return anhs[0].dag() + anhs[0]

q1 = Qubit(
    name = "Q1",
    desc = "Qubit 1",
    comment = "The one and only qubit in this chip",
    freq = 6e9*2*np.pi,
    delta = 100e6 * 2 * np.pi,
    hilbert_dim = 2,
    hamiltonian = duffing,
    )

# r1 = Resonator(
#     name = "R1",
#     desc = "Resonator 1",
#     comment = "The resonator driving Qubit 1",
#     freq = 9e9*2*np.pi,
#     hilbert_dim = 5,
#     hamiltonian = resonator,
#     )
#
# q1r1 = Coupling(
#     name = "Q1-R1",
#     desc = "Coupling between Resonator 1 and Qubit 1",
#     comment = " ",
#     connected = [q1.name, r1.name],
#     strength = 150e6*2*np.pi,
#     hamiltonian = int_XX,
#     )

drive = Drive(
    name = "D1",
    desc = "Drive 1",
    comment = "Drive line 1 on qubit 1 through resonator 1 ",
    connected = [q1.name],
    hamiltonian = drive,
    )

chip_elements = [q1,drive]

initial_model = mdl(chip_elements)
