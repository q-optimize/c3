from test_model import *
from test_generator import *

from c3po.optimizer.optimizer import Optimizer as Opt
from c3po.simulation.simulator import Simulator as Sim
from c3po.utils.tf_utils import *

import copy

rechenknecht = Opt()
rechenknecht.store_history = True

tf_log_level_info()
set_tf_log_level(3)

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
    # 'T_up' : [
    #     (ctrl.get_uuid(), p1.get_uuid())
    #     ],
    # 'T_down' : [
    #     (ctrl.get_uuid(), p1.get_uuid())
    #     ],
    'freq_offset': [(ctrl.get_uuid(), p1.get_uuid())]
}

sim = Sim(initial_model, gen, ctrls)

# Goal to drive on qubit 1
# U_goal = np.array(
#     [[0.+0.j, 1.+0.j, 0.+0.j],
#      [1.+0.j, 0.+0.j, 0.+0.j],
#      [0.+0.j, 0.+0.j, 1.+0.j]]
#     )

U_goal = np.array(
    [[0.+0.j, 1.+0.j],
     [1.+0.j, 0.+0.j]],
    )

sim.model = optimize_model
def evaluate_signals(pulse_params, opt_params):
    model_params = sim.model.params
    U = sim.propagation(pulse_params, opt_params, model_params)
    return 1-tf_unitary_overlap(U, U_goal)

print(
"""
#######################
# Optimizing pulse... #
#######################
"""
)

def callback(xk):
    print(xk)

settings = {'maxiter': 5}

rechenknecht.optimize_controls(
    controls = ctrls,
    opt_map = opt_map,
    opt = 'lbfgs',
#    opt = 'tf_grad_desc',
    settings = settings,
    calib_name = 'openloop',
    eval_func = evaluate_signals,
    callback = callback
    )
