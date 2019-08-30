from test_model import *
from test_generator import *
import copy

from c3po.optimizer.optimizer import Optimizer as Opt
from c3po.simulation.simulator import Simulator as Sim

from tensorflow.python import debug as tf_debug
from c3po.utils.tf_utils import *
opt_map = {
    'amp' : [
        (ctrl.get_uuid(), p1.get_uuid())
        ],
    'T_up' : [
        (ctrl.get_uuid(), p1.get_uuid()),
        ],
    'T_down' : [
        (ctrl.get_uuid(), p1.get_uuid()),
        ],
    'freq_offset': [(ctrl.get_uuid(), p1.get_uuid())]
}

opt_params = ctrls.get_corresponding_control_parameters(opt_map)

sim = Sim(initial_model, gen, ctrls)

values = copy.deepcopy(opt_params['values'])

params = tf.placeholder(
    tf.float64,
    shape=len(values)
    )

ctrls.update_controls(params, opt_params)
gen_output = gen.generate_signals(ctrls.controls)
signals = []
for key in gen_output:
    out = gen_output[key]
    ts = out["ts"]
    signals.append(out["signal"])
dt = tf.cast(ts[1]-ts[0], tf.complex128)
h0, hks = initial_model.get_Hamiltonians()
U_final = sim.propagation(params, opt_params)

sess = tf_setup()
#sess = tf_debug.LocalCLIDebugWrapperSession(sess)

U_final = tf_propagation(h0, hks, signals, dt)
U = sess.run(U_final ,feed_dict={params: values})

indx = initial_model.names.index('Q1')
a_q1 = initial_model.ann_opers[indx]
U_goal = a_q1 + a_q1.dag()

fidelity = tf_unitary_overlap(U_final, U_goal.full())
fid = sess.run(fidelity ,feed_dict={params: values})

grad = tf.gradients(fidelity, params)

gradients = sess.run(grad ,feed_dict={params: values})

values, bounds = ctrls.get_values_bounds(opt_params)
bounds = np.array(bounds)
scale = np.diff(bounds)
gradients[0]*scale.T
