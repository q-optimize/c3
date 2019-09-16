from test_model import *
from test_generator import *
from test_optimization import *

initial_model.update_parameters([5.9e9*2*np.pi,0.9e6*2*np.pi,8.5e9*2*np.pi,148e6*2*np.pi])
sim = Sim(initial_model, gen, ctrls)

def match_model(model_params, measurements):
    model_error = 0
    for m in measurements:
        pulse_params, opt_params = m[0]
        result = m[1]
        U = sim.propagation(pulse_params, opt_params, model_params)
        model_error += tf.abs(
            (1-tf_unitary_overlap(U_goal, U)) - result
            )
    return model_error

rechenknecht.learn_model(
    initial_model,
    eval_func = match_model,
    meas_results = []
    )
