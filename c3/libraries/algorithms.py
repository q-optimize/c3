"""
Collection of (optimization) algorithms. All entries share a common signature with optional arguments.
"""

from scipy.optimize import minimize as minimize
import cma.evolution_strategy as cma
import numpy as np

# from nevergrad.optimization import registry as algo_registry
import adaptive
import copy

algorithms = dict()


def algo_reg_deco(func):
    """
    Decorator for making registry of functions
    """
    algorithms[str(func.__name__)] = func
    return func


@algo_reg_deco
def single_eval(x0, fun=None, fun_grad=None, grad_lookup=None, options={}):
    """
    Return the function value at given point.

    Parameters
    ----------
    x0 : float
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
    fun(x0)


@algo_reg_deco
def grid2D(x0, fun=None, fun_grad=None, grad_lookup=None, options={}):
    """
    Two dimensional scan of the function values around the initial point.

    Parameters
    ----------
    x0 : float
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
    #         probe_list.append(x0)

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
def sweep(x0, fun=None, fun_grad=None, grad_lookup=None, options={}):
    """
    One dimensional scan of the function values around the initial point.

    Parameters
    ----------
    x0 : float
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
            fun([x0[0].numpy()])

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
def adaptive_scan(x0, fun=None, fun_grad=None, grad_lookup=None, options={}):
    """
    One dimensional scan of the function values around the initial point, using adaptive sampling

    Parameters
    ----------
    x0 : float
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
            probe_list.append(x0)

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

    runner = adaptive.runner.simple(
        learner, goal=lambda learner_: learner_.loss() < accuracy_goal
    )


@algo_reg_deco
def lbfgs(x0, fun=None, fun_grad=None, grad_lookup=None, options={}):
    """
    Wrapper for the scipy.optimize.minimize implementation of LBFG-S. See also:

    https://docs.scipy.org/doc/scipy/reference/optimize.minimize-lbfgsb.html

    Parameters
    ----------
    x0 : float
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
    options.update({"disp": True})
    return minimize(fun_grad, x0, jac=grad_lookup, method="L-BFGS-B", options=options)


@algo_reg_deco
def cmaes(x0, fun=None, fun_grad=None, grad_lookup=None, options={}):
    """
    Wrapper for the pycma implementation of CMA-Es. See also:

    http://cma.gforge.inria.fr/apidocs-pycma/

    Parameters
    ----------
    x0 : float
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
            Custom stopping condition. Stop if the cloud shrunk for this number of generations.
        stop_at_sigma : float
            Custom stopping condition. Stop if the cloud shrunk to this standard deviation.

    Returns
    -------
    np.array
        Parameters of the best point.
    """
    custom_stop = False
    if "noise" in options:
        noise = float(options.pop("noise"))
    else:
        noise = 0

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

    es = cma.CMAEvolutionStrategy(x0, spread, settings)
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
                print(f"C3:STATUS:Goal sigma reached. Stopping CMA.")
                break

        samples = es.ask()
        if init_point and iter == 0:
            samples.insert(0, x0)
            print("C3:STATUS:Adding initial point to CMA sample.")
        solutions = []
        for sample in samples:
            goal = fun(sample)
            if noise:
                goal = goal + (np.random.randn() * noise)
            solutions.append(goal)
        es.tell(samples, solutions)
        es.disp()

        iter += 1
    es.result_pretty()
    return es.result.xbest


@algo_reg_deco
def cma_pre_lbfgs(x0, fun=None, fun_grad=None, grad_lookup=None, options={}):
    """
    Performs a CMA-Es optimization and feeds the result into LBFG-S for further refinement.

    """
    x1 = cmaes(x0, fun, options=options["cmaes"])
    lbfgs(x1, fun_grad=fun_grad, grad_lookup=grad_lookup, options=options["lbfgs"])


@algo_reg_deco
def gcmaes(x0, fun=None, fun_grad=None, grad_lookup=None, options={}):
    """
    EXPERIMENTAL CMA-Es where every point in the cloud is optimized with LBFG-S and the resulting cloud and results are
    used for the CMA update.
    """
    custom_stop = False
    options_cma = options["cmaes"]
    if "noise" in options_cma:
        noise = float(options_cma.pop("noise"))
    else:
        noise = 0

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

    es = cma.CMAEvolutionStrategy(x0, spread, settings)
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
                print(f"C3:STATUS:Goal sigma reached. Stopping CMA.")
                break

        samples = es.ask()
        if init_point and iter == 0:
            samples.insert(0, x0)
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


# def oneplusone(x0, goal_fun):
#     optimizer = algo_registry['OnePlusOne'](instrumentation=x0.shape[0])
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
