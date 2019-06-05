"""C3PO configuration file"""

import qutip as qt
from numpy import pi as pi
from c3po.main.gate import Gate as gt
from c3po.utils.envelopes import flattop
"""
Device specific setup goes here. We need to provide a function that takes a
gate, evaluates it in the physical machine and returns a figure of merit.
"""


X_gate = gt('qubit_1', qt.sigmax())


# TODO: deal with freezed parameters

handmade_pulse = {
        'control1': {
            'carrier1': {
                'freq': 6e9*2*pi,
                'pulses': {
                    'pulse1': {
                        'params': {
                            'amp': 15e6*2*pi,
                            't_up': 5e-9,
                            't_down': 45e-9,
                            'xy_angle': 0,
                            'freq_offset': 500e6*2*pi
                            },
                        'func': flattop
                        }
                    }
                }
            }
        }

pulse_bounds = {
        'control1': {
            'carrier1': {
                'pulses': {
                    'pulse1': {
                        'params': {
                            't_up': [2e-9, 98e-9],
                            't_down': [2e-9, 98e-9],
                            'freq_offset': [-1e9*2*pi, 1e9*2*pi]
                            }
                        }
                    }
                }
            }
        }

X_gate.set_parameters('initial', handmade_pulse)
X_gate.set_bounds(pulse_bounds)
