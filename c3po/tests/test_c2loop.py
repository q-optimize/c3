from test_optimization import *

opt_sim = Sim(real_model, gen, ctrls)

def match_model(model_params, opt_params , measurements):
    model_error = 0
    for m in measurements:
        pulse_params = m[0]
        result = m[1]
        U = opt_sim.propagation(pulse_params, opt_params, model_params)
        diff = (1-tf_unitary_overlap(U_goal, U)) - result
        model_error += tf.conj(diff) * diff
    return model_error

settings = {'maxiter': 30}

print(
"""
#######################
# Matching model...   #
#######################
"""
)

rechenknecht.learn_model(
    optimize_model,
    eval_func = match_model,
    settings = settings,
    meas_results = []
    )
