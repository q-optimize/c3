from scipy.optimize import minimize as minimize
import cma.evolution_strategy as cma
import numpy as np
# from nevergrad.optimization import registry as algo_registry


def lbfgs(x0, goal_fun, grad_fun, options={}):
    # TODO print from the log not from hear
    options.update({'disp': True})
    minimize(
        goal_fun,
        x0,
        jac=grad_fun,
        method='L-BFGS-B',
        options=options
    )


def cmaes(x0, goal_fun, options={}):
    if 'noise' in options:
        noise = float(options.pop('noise'))
    if 'init_point' in options:
        init_point = bool(options.pop('init_point'))
    settings = options
    es = cma.CMAEvolutionStrategy(x0, 0.1, settings)
    iter = 0
    while not es.stop():
        samples = es.ask()
        if init_point and iter == 0:
            samples.insert(0,x0)
            print('adding initial point to sample')
        solutions = []
        for sample in samples:
            goal = goal_fun(sample)
            if 'noise' in options:
                goal = goal + (np.random.randn() * noise)
            solutions.append(goal)
        es.tell(samples, solutions)
        es.disp()
        iter += 1


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
