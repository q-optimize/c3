from test_model import *
from test_generator import *

from c3po.optimizer.optimizer import Optimizer as Opt
from c3po.simulation.simulator import Simulator as Sim

from tensorflow.python import debug as tf_debug
from c3po.utils.tf_utils import *

sess = tf_setup()

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

opt_params = ctrls.get_corresponding_control_parameters(opt_map)

sim = Sim(initial_model, gen, ctrls)

values = opt_params['values']

params = tf.placeholder(
    tf.float64,
    shape=len(values)
    )

#U_final = sim.propagation(params, opt_params)

U_final = sim.propagation(params, opt_params)

indx = initial_model.names.index('Q1')
a_q1 = initial_model.ann_opers[indx]
U_goal = a_q1 + a_q1.dag()

fidelity = tf_unitary_overlap(U_final, U_goal.full())

#sess = tf_debug.LocalCLIDebugWrapperSession(sess)

U = sess.run(U_final ,feed_dict={params: values})
