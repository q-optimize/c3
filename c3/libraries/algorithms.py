"""
Collection of (optimization) algorithms. All entries share a common signature with
optional arguments.
"""

from scipy.optimize import minimize as minimize
import cma.evolution_strategy as cma
import numpy as np
import warnings
from typing import Callable
import adaptive
import copy
from scipy.optimize import OptimizeResult
import tensorflow as tf

algorithms = dict()


def algo_reg_deco(func):
    """
    Decorator for making registry of functions
    """
    algorithms[str(func.__name__)] = func
    return func


@algo_reg_deco
def single_eval(x_init, fun=None, fun_grad=None, grad_lookup=None, options={}):
    """
    Return the function value at given point.

    Parameters
    ----------
    x_init : float
        Initial point
    fun : callable
        Goal function
    fun_grad : callable
        Function that computes the gradient of the goal function
    grad_lookup : callable
        Lookup a previously computed gradient
    options : dict
        Algorithm specific options
    """
    fun(x_init)


@algo_reg_deco
def grid2D(x_init, fun=None, fun_grad=None, grad_lookup=None, options={}):
    """
    Two dimensional scan of the function values around the initial point.

    Parameters
    ----------
    x_init : float
        Initial point
    fun : callable
        Goal function
    fun_grad : callable
        Function that computes the gradient of the goal function
    grad_lookup : callable
        Lookup a previously computed gradient
    options : dict
        Options include
        points : int
            The number of samples
        bounds : list
            Range of the scan for both dimensions
    """
    # TODO generalize grid to take any  number of dimensions

    if "points" in options:
        points = options["points"]
    else:
        points = 100

    # probe_list = []
    # if 'probe_list' in options:
    #     for x in options['probe_list']:
    #         probe_list.append(eval(x))

    # if 'init_point' in options:
    #     init_point = bool(options.pop('init_point'))
    #     if init_point:
    #         probe_list.append(x_init)

    bounds = options["bounds"][0]
    bound_min = bounds[0]
    bound_max = bounds[1]
    # probe_list_min = np.min(np.array(probe_list)[:,0])
    # probe_list_max = np.max(np.array(probe_list)[:,0])
    # bound_min = min(bound_min, probe_list_min)
    # bound_max = max(bound_max, probe_list_max)
    xs = np.linspace(bound_min, bound_max, points)

    bounds = options["bounds"][1]
    bound_min = bounds[0]
    bound_max = bounds[1]
    # probe_list_min = np.min(np.array(probe_list)[:,1])
    # probe_list_max = np.max(np.array(probe_list)[:,1])
    # bound_min = min(bound_min, probe_list_min)
    # bound_max = max(bound_max, probe_list_max)
    ys = np.linspace(bound_min, bound_max, points)

    # for p in probe_list:
    #     fun(p)

    for x in xs:
        for y in ys:
            if "wrapper" in options:
                val = copy.deepcopy(options["wrapper"])
                val[val.index("x")] = x
                val[val.index("y")] = y
                fun([val])
            else:
                fun([x, y])


@algo_reg_deco
def sweep(x_init, fun=None, fun_grad=None, grad_lookup=None, options={}):
    """
    One dimensional scan of the function values around the initial point.

    Parameters
    ----------
    x_init : float
        Initial point
    fun : callable
        Goal function
    fun_grad : callable
        Function that computes the gradient of the goal function
    grad_lookup : callable
        Lookup a previously computed gradient
    options : dict
        Options include
        points : int
            The number of samples
        bounds : list
            Range of the scan
    """
    if "points" in options:
        points = options["points"]
    else:
        points = 100

    if "init_point" in options:
        init_point = bool(options["init_point"])
        if init_point:
            fun([x_init[0].numpy()])

    bounds = options["bounds"][0]
    bound_min = bounds[0]
    bound_max = bounds[1]

    probe_list = []
    if "probe_list" in options:
        for x in options["probe_list"]:
            probe_list.append(x)
        probe_list_min = min(probe_list)
        probe_list_max = max(probe_list)
        bound_min = min(bound_min, probe_list_min)
        bound_max = max(bound_max, probe_list_max)

        for p in probe_list:
            if "wrapper" in options:
                val = copy.deepcopy(options["wrapper"])
                val[val.index("x")] = p
                fun([val])
            else:
                fun([p])

    xs = np.linspace(bound_min, bound_max, points)
    for x in xs:
        if "wrapper" in options:
            val = copy.deepcopy(options["wrapper"])
            val[val.index("x")] = x
            fun([val])
        else:
            fun([x])


@algo_reg_deco
def adaptive_scan(x_init, fun=None, fun_grad=None, grad_lookup=None, options={}):
    """
    One dimensional scan of the function values around the initial point, using
    adaptive sampling

    Parameters
    ----------
    x_init : float
        Initial point
    fun : callable
        Goal function
    fun_grad : callable
        Function that computes the gradient of the goal function
    grad_lookup : callable
        Lookup a previously computed gradient
    options : dict
        Options include

        accuracy_goal: float
            Targeted accuracy for the sampling algorithm
        probe_list : list
            Points to definitely include in the sampling
        init_point : boolean
            Include the initial point in the sampling
    """
    if "accuracy_goal" in options:
        accuracy_goal = options["accuracy_goal"]
    else:
        accuracy_goal = 0.5
    print("accuracy_goal: " + str(accuracy_goal))

    probe_list = []
    if "probe_list" in options:
        for x in options["probe_list"]:
            probe_list.append(eval(x))

    if "init_point" in options:
        init_point = bool(options.pop("init_point"))
        if init_point:
            probe_list.append(x_init)

    # TODO make adaptive scan be able to do multidimensional scan
    bounds = options["bounds"][0]
    bound_min = bounds[0]
    bound_max = bounds[1]
    probe_list_min = min(probe_list)
    probe_list_max = max(probe_list)
    bound_min = min(bound_min, probe_list_min)
    bound_max = max(bound_max, probe_list_max)
    print(" ")
    print("bound_min: " + str((bound_min) / (2e9 * np.pi)))
    print("bound_max: " + str((bound_max) / (2e9 * np.pi)))
    print(" ")

    def fun1d(x):
        return fun([x])

    learner = adaptive.Learner1D(fun1d, bounds=(bound_min, bound_max))

    if probe_list:
        for x in probe_list:
            print("from probe_list: " + str(x))
            tmp = learner.function(x)
            print("done\n")
            learner.tell(x, tmp)

    adaptive.runner.simple(
        learner, goal=lambda learner_: learner_.loss() < accuracy_goal
    )


@algo_reg_deco
def tf_sgd(
    x_init: np.array,
    fun: Callable = None,
    fun_grad: Callable = None,
    grad_lookup: Callable = None,
    options: dict = {},
) -> OptimizeResult:
    """Optimize using TensorFlow Stochastic Gradient Descent with Momentum
    https://www.tensorflow.org/api_docs/python/tf/keras/optimizers/SGD

    Parameters
    ----------
    x_init : np.array
        starting value of parameter(s)
    fun : Callable, optional
        function to minimize, by default None
    fun_grad : Callable, optional
        gradient of function to minimize, by default None
    grad_lookup : Callable, optional
        lookup stored gradients, by default None
    options : dict, optional
        optional parameters for optimizer, by default {}

    Returns
    -------
    OptimizeResult
        SciPy OptimizeResult type object with final parameters
    """

    if "maxfun" in options.keys():
        raise KeyError("Tensorflow Optimizers require a maxiters")

    iters = options["maxiters"]  # TF based optimizers use algo iters not fevals

    var = tf.Variable(x_init)

    def tf_fun():
        return fun(var)

    opt_sgd = tf.keras.optimizers.SGD(learning_rate=0.1, momentum=0.9)

    for step in range(iters):
        step_count = opt_sgd.minimize(tf_fun, [var])
        print(f"epoch {step_count.numpy()}: func_value: {tf_fun()}")

    result = OptimizeResult(x=var.numpy(), success=True)
    return result


@algo_reg_deco
def tf_adam(
    x_init: np.array,
    fun: Callable = None,
    fun_grad: Callable = None,
    grad_lookup: Callable = None,
    options: dict = {},
) -> OptimizeResult:
    """Optimize using TensorFlow ADAM
    https://www.tensorflow.org/api_docs/python/tf/keras/optimizers/Adam

    Parameters
    ----------
    x_init : np.array
        starting value of parameter(s)
    fun : Callable, optional
        function to minimize, by default None
    fun_grad : Callable, optional
        gradient of function to minimize, by default None
    grad_lookup : Callable, optional
        lookup stored gradients, by default None
    options : dict, optional
        optional parameters for optimizer, by default {}

    Returns
    -------
    OptimizeResult
        SciPy OptimizeResult type object with final parameters
    """
    # TODO Update maxfun->maxiters, default hyperparameters and error handling
    warnings.warn("The integration of this algorithm is incomplete and incorrect.")

    iters = options["maxfun"]
    var = tf.Variable(x_init)

    def tf_fun():
        return fun(var)

    opt_adam = tf.keras.optimizers.Adam(learning_rate=0.001, epsilon=0.1)

    for step in range(iters):
        step_count = opt_adam.minimize(tf_fun, [var])
        print(f"epoch {step_count.numpy()}: func_value: {tf_fun()}")

    result = OptimizeResult(x=var.numpy(), success=True)
    return result


def tf_rmsprop(
    x_init: np.array,
    fun: Callable = None,
    fun_grad: Callable = None,
    grad_lookup: Callable = None,
    options: dict = {},
) -> OptimizeResult:
    """Optimize using TensorFlow RMSProp
    https://www.tensorflow.org/api_docs/python/tf/keras/optimizers/RMSprop

    Parameters
    ----------
    x_init : np.array
        starting value of parameter(s)
    fun : Callable, optional
        function to minimize, by default None
    fun_grad : Callable, optional
        gradient of function to minimize, by default None
    grad_lookup : Callable, optional
        lookup stored gradients, by default None
    options : dict, optional
        optional parameters for optimizer, by default {}

    Returns
    -------
    OptimizeResult
        SciPy OptimizeResult type object with final parameters
    """
    # TODO Update maxfun->maxiters, default hyperparameters and error handling
    warnings.warn("The integration of this algorithm is incomplete and incorrect.")

    iters = options["maxfun"]

    var = tf.Variable(x_init)

    def tf_fun():
        return fun(var)

    opt_rmsprop = tf.keras.optimizers.RMSprop(
        learning_rate=0.1, epsilon=1e-2, centered=True
    )

    for step in range(iters):
        step_count = opt_rmsprop.minimize(tf_fun, [var])
        print(f"epoch {step_count.numpy()}: func_value: {tf_fun()}")

    result = OptimizeResult(x=var.numpy(), success=True)
    return result


@algo_reg_deco
def tf_adadelta(
    x_init: np.array,
    fun: Callable = None,
    fun_grad: Callable = None,
    grad_lookup: Callable = None,
    options: dict = {},
) -> OptimizeResult:
    """Optimize using TensorFlow Adadelta
    https://www.tensorflow.org/api_docs/python/tf/keras/optimizers/Adadelta

    Parameters
    ----------
    x_init : np.array
        starting value of parameter(s)
    fun : Callable, optional
        function to minimize, by default None
    fun_grad : Callable, optional
        gradient of function to minimize, by default None
    grad_lookup : Callable, optional
        lookup stored gradients, by default None
    options : dict, optional
        optional parameters for optimizer, by default {}

    Returns
    -------
    OptimizeResult
        SciPy OptimizeResult type object with final parameters
    """
    # TODO Update maxfun->maxiters, default hyperparameters and error handling
    warnings.warn("The integration of this algorithm is incomplete and incorrect.")

    iters = options["maxfun"]

    var = tf.Variable(x_init)

    def tf_fun():
        return fun(var)

    opt_adadelta = tf.keras.optimizers.Adadelta(
        learning_rate=0.1, rho=0.95, epsilon=1e-2
    )

    for step in range(iters):
        step_count = opt_adadelta.minimize(tf_fun, [var])
        print(f"epoch {step_count.numpy()}: func_value: {tf_fun()}")

    result = OptimizeResult(x=var.numpy(), success=True)
    return result


@algo_reg_deco
def lbfgs(x_init, fun=None, fun_grad=None, grad_lookup=None, options={}):
    """
    Wrapper for the scipy.optimize.minimize implementation of LBFG-S. See also:

    https://docs.scipy.org/doc/scipy/reference/optimize.minimize-lbfgsb.html

    Parameters
    ----------
    x_init : float
        Initial point
    fun : callable
        Goal function
    fun_grad : callable
        Function that computes the gradient of the goal function
    grad_lookup : callable
        Lookup a previously computed gradient
    options : dict
        Options of scipy.optimize.minimize

    Returns
    -------
    Result
        Scipy result object.
    """
    # TODO print from the log not from here
    # options.update({"disp": True})
    return minimize(
        fun_grad, x_init, jac=grad_lookup, method="L-BFGS-B", options=options
    )


@algo_reg_deco
def lbfgs_grad_free(x_init, fun=None, fun_grad=None, grad_lookup=None, options={}):
    """
    Wrapper for the scipy.optimize.minimize implementation of LBFG-S.
    We let the algorithm determine the gradient by its own.
     See also:

    https://docs.scipy.org/doc/scipy/reference/optimize.minimize-lbfgsb.html

    Parameters
    ----------
    x_init : float
        Initial point
    fun : callable
        Goal function
    fun_grad : callable
        Function that computes the gradient of the goal function
    grad_lookup : callable
        Lookup a previously computed gradient
    options : dict
        Options of scipy.optimize.minimize

    Returns
    -------
    Result
        Scipy result object.
    """
    return minimize(fun=fun, x0=x_init, options=options)


@algo_reg_deco
def cmaes(x_init, fun=None, fun_grad=None, grad_lookup=None, options={}):
    """
    Wrapper for the pycma implementation of CMA-Es. See also:

    http://cma.gforge.inria.fr/apidocs-pycma/

    Parameters
    ----------
    x_init : float
        Initial point.
    fun : callable
        Goal function.
    fun_grad : callable
        Function that computes the gradient of the goal function.
    grad_lookup : callable
        Lookup a previously computed gradient.
    options : dict
        Options of pycma and the following custom options.

        noise : float
            Artificial noise added to a function evaluation.
        init_point : boolean
            Force the use of the initial point in the first generation.
        spread : float
            Adjust the parameter spread of the first generation cloud.
        stop_at_convergence : int
            Custom stopping condition. Stop if the cloud shrunk for this number of
            generations.
        stop_at_sigma : float
            Custom stopping condition. Stop if the cloud shrunk to this standard
            deviation.

    Returns
    -------
    np.array
        Parameters of the best point.
    """
    if "noise" in options:
        noise = float(options.pop("noise"))
    else:
        noise = 0

    if "batch_noise" in options:
        batch_noise = float(options.pop("batch_noise"))
    else:
        batch_noise = 0

    if "init_point" in options:
        init_point = bool(options.pop("init_point"))
    else:
        init_point = False

    if "spread" in options:
        spread = float(options.pop("spread"))
    else:
        spread = 0.1

    shrunk_check = False
    if "stop_at_convergence" in options:
        sigma_conv = int(options.pop("stop_at_convergence"))
        sigmas = []
        shrunk_check = True

    sigma_check = False
    if "stop_at_sigma" in options:
        stop_sigma = int(options.pop("stop_at_sigma"))
        sigma_check = True

    settings = options

    es = cma.CMAEvolutionStrategy(x_init, spread, settings)
    iter = 0
    while not es.stop():

        if shrunk_check:
            sigmas.append(es.sigma)
            if iter > sigma_conv:
                if all(
                    sigmas[-(i + 1)] < sigmas[-(i + 2)] for i in range(sigma_conv - 1)
                ):
                    print(
                        f"C3:STATUS:Shrunk cloud for {sigma_conv} steps. "
                        "Switching to gradients."
                    )
                    break

        if sigma_check:
            if es.sigma < stop_sigma:
                print("C3:STATUS:Goal sigma reached. Stopping CMA.")
                break

        samples = es.ask()
        if init_point and iter == 0:
            samples.insert(0, x_init)
            print("C3:STATUS:Adding initial point to CMA sample.")
        solutions = []
        if batch_noise:
            error = np.random.randn() * noise
        for sample in samples:
            goal = fun(sample)
            if noise:
                error = np.random.randn() * noise
            if batch_noise or noise:
                goal = goal + error
            solutions.append(goal)
        es.tell(samples, solutions)
        es.disp()

        iter += 1
    es.result_pretty()
    return es.result.xbest


@algo_reg_deco
def cma_pre_lbfgs(x_init, fun=None, fun_grad=None, grad_lookup=None, options={}):
    """
    Performs a CMA-Es optimization and feeds the result into LBFG-S for further
    refinement.

    """
    if "cmaes" not in options:
        options["cmaes"] = {}
    if "lbfgs" not in options:
        options["lbfgs"] = {}
    for k in options:
        if k == "cmaes" or k == "lbfgs":
            continue
        else:
            if k not in options["cmaes"]:
                options["cmaes"][k] = options[k]
            if k not in options["lbfgs"]:
                options["lbfgs"][k] = options[k]

    x1 = cmaes(x_init, fun, options=options["cmaes"])
    lbfgs(x1, fun_grad=fun_grad, grad_lookup=grad_lookup, options=options["lbfgs"])


@algo_reg_deco
def gcmaes(x_init, fun=None, fun_grad=None, grad_lookup=None, options={}):
    """
    EXPERIMENTAL CMA-Es where every point in the cloud is optimized with LBFG-S and the
    resulting cloud and results are used for the CMA update.
    """
    options_cma = options["cmaes"]

    if "init_point" in options_cma:
        init_point = bool(options_cma.pop("init_point"))
    else:
        init_point = False

    if "spread" in options_cma:
        spread = float(options_cma.pop("spread"))
    else:
        spread = 0.1

    shrinked_check = False
    if "stop_at_convergence" in options_cma:
        sigma_conv = int(options_cma.pop("stop_at_convergence"))
        sigmas = []
        shrinked_check = True

    sigma_check = False
    if "stop_at_sigma" in options_cma:
        stop_sigma = int(options_cma.pop("stop_at_sigma"))
        sigma_check = True

    settings = options_cma

    es = cma.CMAEvolutionStrategy(x_init, spread, settings)
    iter = 0
    while not es.stop():

        if shrinked_check:
            sigmas.append(es.sigma)
            if iter > sigma_conv:
                if all(
                    sigmas[-(i + 1)] < sigmas[-(i + 2)] for i in range(sigma_conv - 1)
                ):
                    print(
                        f"C3:STATUS:Shrinked cloud for {sigma_conv} steps. "
                        "Switching to gradients."
                    )
                    break

        if sigma_check:
            if es.sigma < stop_sigma:
                print("C3:STATUS:Goal sigma reached. Stopping CMA.")
                break

        samples = es.ask()
        if init_point and iter == 0:
            samples.insert(0, x_init)
            print("C3:STATUS:Adding initial point to CMA sample.")
        solutions = []
        points = []
        for sample in samples:
            r = lbfgs(
                sample,
                fun_grad=fun_grad,
                grad_lookup=grad_lookup,
                options=options["lbfgs"],
            )
            solutions.append(r.fun)
            points.append(r.x)
        es.tell(points, solutions)
        es.disp()

        iter += 1
    return es.result.xbest


# def oneplusone(x_init, goal_fun):
#     optimizer = algo_registry['OnePlusOne'](instrumentation=x_init.shape[0])
#     while True:
#         # TODO make this logging happen elsewhere
#         # self.logfile.write(f"Batch {self.evaluation}\n")
#         # self.logfile.flush()
#         tmp = optimizer.ask()
#         samples = tmp.args
#         solutions = []
#         for sample in samples:
#             goal = goal_fun(sample)
#             solutions.append(goal)
#         optimizer.tell(tmp, solutions)
#
#     # TODO deal with storing best value elsewhere
#     # recommendation = optimizer.provide_recommendation()
#     # return recommendation.args[0]
