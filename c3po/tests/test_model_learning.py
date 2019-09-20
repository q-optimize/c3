from test_model import *
from test_generator import *

from c3po.optimizer.optimizer import Optimizer as Opt
from c3po.simulation.simulator import Simulator as Sim
from c3po.utils.tf_utils import *

rechenknecht = Opt()

tf_log_level_info()

set_tf_log_level(2)

print("current log level: " + str(get_tf_log_level()))

# to make sure session is empty look up: tf.reset_default_graph()

sess = tf_setup()

print(" ")
print("Available tensorflow devices: ")
tf_list_avail_devices()
writer = tf.summary.FileWriter( '/tmp/tf_logs/optim_log', sess.graph)
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

initial_model.update_parameters([5.9e9*2*np.pi,0.9e6*2*np.pi,8.5e9*2*np.pi,148e6*2*np.pi])
sim = Sim(initial_model, gen, ctrls)

indx = initial_model.names.index('Q1')
a_q1 = initial_model.ann_opers[indx]
U_goal = a_q1 + tf.transpose(a_q1)
opt_params = ctrls.get_corresponding_control_parameters(opt_map)
pulse_params, bounds = ctrls.get_values_bounds(opt_params)
stored_measurement = [[pulse_params, opt_params], [U_goal, 0.40053653853211163]]

def match_model(model_params, opt_params , measurements):
    model_error = 0
    for m in measurements[-5::]:
        pulse_params = m[0]
        result = m[1]
        U = sim.propagation(pulse_params, opt_params, model_params)
        U = opt_sim.propagation(pulse_params, opt_params, model_params)
        diff = (1-tf_unitary_overlap(U_goal, U)) - result
        model_error += tf.conj(diff) * diff
    return model_error

rechenknecht.learn_model(
    initial_model,
    eval_func = match_model,
    meas_results = [stored_measurement]
    )
