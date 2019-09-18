"""C3PO configuration file"""

import numpy as np
import qutip as qt
import tensorflow as tf
import matplotlib.pyplot as plt

import c3po

from c3po.cobj.component import *
from c3po.main.model import Model as mdl

import time

from tensorflow.python import debug as tf_debug

q1 = Qubit(
    name = "Q1",
    desc = "Qubit 1",
    comment = "The one and only qubit in this chip",
    freq = 6e9*2*np.pi,
    delta = 1e6 * 2 * np.pi,
    hilbert_dim = 2
    )

r1 = Resonator(
    name = "R1",
    desc = "Resonator 1",
    comment = "The resonator driving Qubit 1",
    freq = 9e9*2*np.pi,
    hilbert_dim = 5
    )

q1r1 = Coupling(
    name = "Q1-R1",
    desc = "Coupling between Resonator 1 and Qubit 1",
    comment = " ",
    connected = [q1.name, r1.name],
    strength = 150e6*2*np.pi
    )

drive = Drive(
    name = "D1",
    desc = "Drive 1",
    comment = "Drive line 1 on qubit 1",
    connected = [q1.name]
    )

chip_elements = [
    q1,
     # r1,
     # q1r1,
     drive
     ]

initial_model = mdl(chip_elements)
