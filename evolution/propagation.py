
# REMARK: Until a clearer specification of the way the hamiltonian is passed
#         is given, i.e. what is the exact return value/format of
#         get_Hamiltonian of class model this part stays quiet vague.


from qutip import Qobj
from qutip import sesolve

from c3po.utils import tf_utils

import tensorflow as tf
import numpy as np

def propergate(model, u0, tlist, method, grad = False, history = False):
    """
    Wrapper function for choosing the type of propergation method.
    :param model:   :class:'c3po.model'
        Meta class for system hamiltonian
    :param u0:
        can be a state, a unitary or ...
    :param tlist:
        ...
    :param  method:
        switch for choosing the desired propergator. options are: "pwc",
        "pwc_tf", "qutip_sesolv"

    """
    if method == "pwc":
        H = model.get_Hamiltonian() # Hamilton as lambda func or list [H0,...]?
        tmp = Qobj      # create temp obj to get access to method evalute
        H = tmp.evaluate(H) # from qutip: Evaluate a time-dependent quantum 
                            # object in list format.
                            # see: http://qutip.org/docs/latest/apidoc/classes.html?highlight=evaluate#qutip.Qobj.evaluate
        U = sesolve_pwc(H, u0, tlist, grad, history)

    if method == "pwc_tf":
        H = model.get_Hamiltonian_tf()
        H = tf_evaluate(H)
        # u0 must also be converted to tensorflow
        U = sesolve_pwc_tf(H, u0, tlist, grad, history)

    if method == "qutip_sesolv":
        H = model.get_Hamiltonian()
        U = sesolve(H, u0, tlist)

    return U


def sesolve_pwc(H, u0, tlist, grad = False, history = False):
    """
    Find the propagator of a system Hamiltonian H(t). The initial basis u0. The
    hamiltonian
    :param H:   :class:'qutip.qobj'
        System hamiltonian or a list of Drift and Control terms
    :param u0:
        can be a state, a unitary or a rectangular matrix made of several
        initial states.
    :param tlist:
        time vector
    :output:    list of unitaries (Qobj)
        for all times in tlist or just the initial and final time (depending
        on history setting)
    """


# CODE TAKEN FROM FEDERICO'S CODE: new_IBM_USAAR/utils.py

    if history:
        U = [u0]

        for t in tlist[1::]:
            dU = (-1j * dt * H(t+dt/2)).expm()
            U.append(dU * U[-1])

    else:
        U = [u0]

        for t in tlist[1::]:
            dU = (-1j * dt * H(t+dt/2)).expm()
            U[0] = dU * U[0]


    return U


def sesolve_pwc_tf(H, u0, tlist, grad = False, history = False):
    """
    Find the propagator of a system Hamiltonian H(t). The initial basis u0. The
    hamiltonian
    :param H:  given in tensorflow
        System hamiltonian or a list of Drift and Control terms
    :param u0: must also be given in tensorflow(!)
        can be a state, a unitary or a rectangular matrix made of several
        initial states.
    :param tlist:
        time vector
    :output:    list of unitaries (Qobj)
        for all times in tlist or just the initial and final time (depending
        on history setting)
    """


# CODE TAKEN FROM NICO'S CODE: tf_propagation.py

    U = 0

    return U



