"""
A library of fidelity functions and combiners.
"""

import numpy as np

def combine_goal_penalty_smooth(
        goal_error,
        penalty,
        g_thresh,
        p_thresh,
        opts={
            'weight of sum': 2,
            'exp of sum': 2,
            'weight of difference': 1,
            'exp of difference': 1
            }
    ):
    """
    Combine a fidelity with penalties. You can specify parameters for the
    combination in the following way:
    result = c1(g+p)^e1 + c2(g-p)^e2
    Default values shown below:
    opts = {
        'weight of sum': 2,
        'exp of sum': 2,
        'weight of difference': 1,
        'exp of difference': 1
    }
    """
    c_sum = opts['weight of sum']
    e_sum = opts['exp of sum']
    c_diff = opts['weight of difference']
    e_diff = opts['exp of difference']

    g_eff = goal_error - g_thresh
    p_eff = penalty - p_thresh

    gq_sum = c_sum * (g_eff + p_eff)**e_sum
    gq_diff = c_diff * np.abs(g_eff - p_eff)**e_diff

    return np.log10(gq_sum + gq_diff)
