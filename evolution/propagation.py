def sesolve_pwc(H, u0, tlist, grad = False, history = False):
    """
    Find the propagator of a system Hamiltonian h(t). The initial basis u0  The hamiltonian
    :param H: :class:'qutip.qobj' System hamiltonian or a list of Drift and Control terms
    :param u0: can be a state, a unitary or a rectangular matrix made of several initial states.
    :param tlist: time vector
    :output: list of unitaries (Qobj) for all times in tlist or just the initial and final time (depending on history setting)
    """
    
    us_of_t = []
    
    return us_of_t
