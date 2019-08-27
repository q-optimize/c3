from test_generator import *
from test_model import *

from c3po.optimizer.optimizer import Optimizer as Opt
from c3po.simulation.simulator import Simulator as Sim

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

opt_params = ctrls.get_corresponding_control_parameters(opt_map)

sim = Sim(initial_model, gen, ctrls)

values = opt_params['values']

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
#U_final = sim.propagation(params, opt_params)

U_final = tf_propagation(h0, hks, signals, dt)

U = sess.run(U_final ,feed_dict={params: values})
