"""C3PO test file"""

import qutip as qt
import numpy as np
from numpy import pi as pi
from c3po.main.gate import Gate as gt
from c3po.utils.envelopes import flattop
from c3po.utils.tf_utils import tf_setup
"""
Device specific setup goes here. We need to provide a function that takes a
gate, evaluates it in the physical machine and returns a figure of merit.
"""


X_gate = gt('qubit_1', qt.sigmax())

######################
# Test linear params #
######################

params = {
    'A': 5,
    'L': 5,
    'c':2
    }

bounds = {
    'A': [1,5],
#    'L': [1,5],
    'c':[1,2]
    }
tf_sess = tf_setup()

X_gate.env_shape = 'flat'
X_gate.set_parameters('initial', params)
X_gate.set_bounds(bounds)
print(X_gate.idxes)
print(X_gate.opt_idxes)
print(tf_sess.run(X_gate.to_scale_one('initial')))
