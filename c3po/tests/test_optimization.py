from test_model import *
from test_generator import *

from c3po.optimizer.optimizer import Optimizer as Opt
from c3po.simulation.simulator import Simulator as Sim
from c3po.utils.tf_utils import *

import copy

rechenknecht = Opt()

tf_log_level_info()

set_tf_log_level(2)

print("current log level: " + str(get_tf_log_level()))

# to make sure session is empty look up: tf.reset_default_graph()

sess = tf_setup()

print(" ")
print("Available tensorflow devices: ")
tf_list_avail_devices()
writer = tf.summary.FileWriter( './logs/optim_log', sess.graph)
rechenknecht.set_session(sess)
rechenknecht.set_log_writer(writer)

opt_map = {
    'amp' : [
        (ctrl.get_uuid(), p1.get_uuid())
        ],
    'T_up' : [
        (ctrl.get_uuid(), p1.get_uuid())
        ],
    'T_down' : [
        (ctrl.get_uuid(), p1.get_uuid())
        ],
    'freq_offset': [(ctrl.get_uuid(), p1.get_uuid())]
}

ctrls_guess = copy.deepcopy(ctrls)

sim = Sim(initial_model, gen, ctrls)

# Goal to drive on qubit 1
indx = initial_model.names.index('Q1')
a_q1 = initial_model.ann_opers[indx]
U_goal = a_q1 + tf.transpose(a_q1)

def evaluate_signals(pulse_params, opt_params):
    model_params = sim.model.params
    U = sim.propagation(pulse_params, opt_params, model_params)
    return 1-tf_unitary_overlap(U, U_goal)

rechenknecht.optimize_controls(
    controls = ctrls,
    opt_map = opt_map,
    opt = 'lbfgs',
#    opt = 'tf_grad_desc',
    settings=None,
    calib_name = 'test',
    eval_func = evaluate_signals
    )

ctrls_openloop = copy.deepcopy(ctrls)
