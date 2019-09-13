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

sim = Sim(initial_model, gen, ctrls)


def match_model(model_params, measurements):
    model_error = 0

    for m in measurements:
        pulse_params, opt_params = m[0]
        operator, result = m[1]
        U = sim.propagation(model_params, pulse_params, opt_params)
        model_error += tf.abs(
            tf_measure_operator(operator, U) - result
            )

    return model_error

indx = initial_model.names.index('Q1')
a_q1 = initial_model.ann_opers[indx]
measured_op = a_q1 + tf.transpose(a_q1)

opt_params = ctrls.get_corresponding_control_parameters(opt_map)
pulse_params, bounds = ctrls.get_values_bounds(opt_params)

stored_measurement = [
    [[pulse_params, opt_params], [measured_op, 3]]
]

rechenknecht.learn_model(
    initial_model,
    eval_func = match_model,
    meas_results = stored_measurement
    )
