
# REMARK: Until a clearer specification of the way the hamiltonian is passed
#         is given, i.e. what is the exact return value/format of
#         get_Hamiltonian of class model this part stays quiet vague.

#
# !!! CAUTION !!!
# None of this code has been tested yet!


from qutip import Qobj
from qutip import sesolve

from c3po.utils import tf_utils

import tensorflow as tf
import numpy as np

def propagate(model, gate, u0, tlist, method, grad = False, history = False):
    """
    Wrapper function for choosing the type of propagation method.
    :param model:   :class:'c3po.model'
        Meta class for system hamiltonian
    :param u0:
        can be a state, a unitary or ...
    :param tlist:
        ...
    :param  method:
        switch for choosing the desired propagator. options are: "pwc",
        "pwc_tf", "qutip_sesolv"

    """



    if method == "pwc":
        H = model.get_Hamiltonian()

        params = gate.get_parameters() # retrieve parameters/args for the drive
                                       # fields

        tmp = Qobj      # create temp obj to get access to method evalute
        H = tmp.evaluate(H, t, params) # from qutip: Evaluate a time-dependent quantum 
                            # object in list format.
                            # see: http://qutip.org/docs/latest/apidoc/classes.html?highlight=evaluate#qutip.Qobj.evaluate
        U = sesolve_pwc(H, u0, tlist, grad, history)

    if method == "pwc_tf":

        # !!! CAUTION !!!
        # this setup (tf_setup) step needs to be moved!
        # I assume that you cannot create tensorflow objects without any 
        # previously created session, as I guess that tf-objects are tied 
        # to the initialized session. This means that everytime the model-class
        # tries to return an Hamilton as tf-object a tf-session should be 
        # running
        # Right now I just put this here to remind, that a session 
        # has to be created in order to use tensorflow, but this is clearly
        # not the right place to do it. 
        # My guess: Session initialization should be done at the 
        # very beginning, when systems/models/problems are specified/configured

        H = model.get_Hamiltonian_tf()  # should return Hamilton as lambda func 
                                        # or list [H0, ...]


        H = tf_evaluate(H)  # needed to convert a list [H0, ...] in useable format


        params = model.get_params_tf()  # system parameters should be provided 
                                        # by model for the tensorflow backend
                                        # as input for: 
                                        # session.run(..., feed_dict = params)


        # u0 must also be converted to tensorflow, should u0 be part of params?
        # this would make sense and streamline code imo.
        U = sesolve_pwc_tf(H, params, tlist, sess, grad, history)

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


def sesolve_pwc_tf(H, params, tlist, session, grad = False, history = False):
    """
    Find the propagator of a system Hamiltonian H(t). The initial basis u0. The
    hamiltonian
    :param H:  given in tensorflow
        System hamiltonian or a list of Drift and Control terms
    """


# CODE TAKEN FROM NICO'S CODE: tf_propagation.py
#
# !!! CAUTION/REMARK !!!
# I do not expect this code to run. I still have to read more about 
# tensorflow. I put this code here as an overview for myself(Kevin) and others
# Please see this code as placeholder.



    # convert tlist to tensorflor object
    # this is just a hack for now. and hasn't been tested but should 
    # produce some working code. Regard it as placeholder.
    Ts = tf.linspace(tlist[0], tlist[len(tlist - 1)], len(tlist), name="Time")
    Ts = tf.to_complex64(Ts)


    # with tf.name_scope('U_actual'):
        # def condition(i, u):
            # return i < N_slices

        # def body(i, u):
            # t = Ts[i] + dt/2
            # u_ = tf.linalg.expm(-1j / hbar * (H(t) * dt) * u
            # return i+1, u_

        # max_i, uf = tf.while_loop(condition, body, (0, u_initial), parallel_interactions=10)

    # with session.as_default():
        # run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
        # run_metadata = tf.RunMetadata()
        # session.run(H, feed_dict=params)

    U = 0

    return U



