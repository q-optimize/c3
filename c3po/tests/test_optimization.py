from test_generator import *
from test_model import *

from c3po.optimizer.optimizer import Optimizer as Opt
from c3po.simulation.simulator import Simulator as Sim

rechenknecht = Opt()

opt_map = {
    'amp' : [
        (ctrl.get_uuid(), p1.get_uuid()),
        (ctrl.get_uuid(), p2.get_uuid())
        ],
    'T_up' : [
        (ctrl.get_uuid(), p1.get_uuid()),
        (ctrl.get_uuid(), p2.get_uuid())
        ],
    'T_down' : [
        (ctrl.get_uuid(), p1.get_uuid()),
        (ctrl.get_uuid(), p2.get_uuid())
        ],
    'xy_angle': [(ctrl.get_uuid(), p2.get_uuid())],
    'freq_offset': [(ctrl.get_uuid(), p1.get_uuid())]
}

import pprint
pprint.pprint(opt_map)
opt_params = rechenknecht.get_corresponding_signal_parameters([ctrl], opt_map)
pprint.pprint(opt_params)

sim = Sim()

def evaluate_signals(U):
    U = rechenknecht.propagation(params)
    return 1-tf_unitary_overlap(U, U_goal)

rechenknecht.optimize_gate(
    signals = [ctrl],
    opt_map = opt_map,
    opt = 'open_loop',
    settings = opt_settings,
    calib_name = 'test',
    eval_func = evaluate_signals
    )
