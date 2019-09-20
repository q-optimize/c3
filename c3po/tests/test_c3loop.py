from test_optimization import *

rechenknecht.optimizer_history = []

exp_sim = Sim(real_model, gen, ctrls)

def experiment_evaluate(pulse_params, opt_params):
    model_params = exp_sim.model.params
    U = exp_sim.propagation(pulse_params, opt_params, model_params)
    psi_actual = tf.matmul(U, psi_init)
    overlap = tf.matmul(psi_goal.T, psi_actual)
    return 1-tf.cast(tf.conj(overlap)*overlap, tf.float64)


initial_spread = [5e6*2*np.pi, 20e6*2*np.pi]

opt_settings = {
    'CMA_stds': initial_spread,
#    'maxiter' : 20,
    'ftarget' : 1e-4,
    'popsize' : 20
}

print(
"""
#######################
# Calibrating...      #
#######################
"""

)

rechenknecht.optimize_controls(
    controls = ctrls,
    opt_map = opt_map,
    opt = 'cmaes',
#    opt = 'tf_grad_desc',
    settings = opt_settings,
    calib_name = 'closedloop',
    eval_func = experiment_evaluate
    )

opt_sim = Sim(real_model, gen, ctrls)

def match_model_U(model_params, opt_params, measurements):
    model_error = 0
    measurements = measurements[-100::5]
    for m in measurements:
        pulse_params = m[0]
        result = m[1]
        U = opt_sim.propagation(pulse_params, opt_params, model_params)
        diff = (1-tf_unitary_overlap(U_goal, U)) - result
        model_error += tf.conj(diff) * diff
    return model_error

def match_model(model_params, opt_params, measurements):
    model_error = 0
    measurements = measurements[-5::]
    for m in measurements:
        pulse_params = m[0]
        result = m[1]
        U = sim.propagation(pulse_params, opt_params, model_params)
        psi_actual = tf.matmul(U, psi_init)
        overlap = tf.matmul(psi_goal.T, psi_actual)
        diff = (1-tf.cast(tf.conj(overlap)*overlap, tf.float64)) - result
        model_error += diff * diff
    return model_error

def match_model_psi(model_params, opt_params, measurements):
    model_error = 0
    measurements = measurements[-5::]
    for m in measurements:
        pulse_params = m[0]
        result = m[1]
        U = sim.propagation(pulse_params, opt_params, model_params)
        psi_actual = tf.matmul(U, psi_init)
        overlap = tf.matmul(psi_goal.T, psi_actual)
        diff = (1-tf.cast(tf.conj(overlap)*overlap, tf.float64)) - result
        model_error += diff * diff
    return model_error


settings = {'maxiter': 100}

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

tf.summary.FileWriter( '/tmp/c3_logs', sess.graph)
