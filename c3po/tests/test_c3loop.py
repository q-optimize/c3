from test_optimization import *

rechenknecht.optimizer_history = []

real_model = copy.deepcopy(initial_model)
real_model.update_parameters([5.8e9*2*np.pi,0.7e6*2*np.pi,8.5e9*2*np.pi,143e6*2*np.pi])
exp_sim = Sim(real_model, gen, ctrls)

def experiment_evaluate(pulse_params, opt_params):
    model_params = exp_sim.model.params
    U = exp_sim.propagation(pulse_params, opt_params, model_params)
    return 1-tf_unitary_overlap(U, U_goal)

rechenknecht.optimize_controls(
    controls = ctrls,
    opt_map = opt_map,
    opt = 'cmaes',
#    opt = 'tf_grad_desc',
    settings=None,
    calib_name = 'test',
    eval_func = experiment_evaluate
    )

ctrls_closedloop = copy.deepcopy(ctrls)

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
