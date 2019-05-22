
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

def dirty_wrap(func):
    return lambda t, args: func(t)


def propagate(model, gate, u0, tlist, method, tf_sess = None, grad = False, history = False):
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
    methods = ["pwc", "pwc_tf", "qutip_sesolve"]

    if method in methods:
        if "tf" not in method:


            # this is garbage and needs to be fixed. either get_control_fields
            # delivers a list of all control fields or only a function 
            # that is added here to a list
            keys = gate.get_parameters().keys()
            control_fields = []
            for key in keys:
                org_func = gate.get_control_fields(key)
                control_fields.append(dirty_wrap(org_func))

            # dictrionary of parameters for crontrol fields
            # but this is kind of obsolete as sesolve won't need arguments(?)
            params = gate.get_parameters()

            # should return Hamilton as list [H0, ...]
            hlist = model.get_Hamiltonian(control_fields)
        else:
            control_fields = gate.get_tf_control_fields()

            params = gate.get_parameters()  # retrieve parameters/args for the drive
                                            # fields

            params.update(model.get_params_tf())    # system parameters should be provided 
                                                    # by model for the tensorflow backend
                                                    # as input for: 
                                                    # session.run(..., feed_dict = params)

            # params.update(u0) # u0 should be part of the params passed to 
                                # tensorflow and also needs to be a initialized/converted
                                # as tensorflow object

            hlist = model.get_tf_Hamiltonian(control_fields)


        if method == "pwc":
            U = sesolve_pwc(hlist, u0, tlist, params, grad, history="True")

        if method == "pwc_tf":
            U = sesolve_pwc_tf(hlist, params, tlist, tf_sess, grad, history)

        if method == "qutip_sesolve":
#             dim = u0.shape[1]
            # U = []
            # for i in range(0, dim):
                # tmp = Qobj()
                # U.append(sesolve(hlist, u0[i], tlist))
            U = sesolve(hlist, u0, tlist)

    return U


def sesolve_pwc(H, u0, tlist, args={}, grad = False, history = False):
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

#     if history:
        # U = [u0]

        # for t in tlist[1::]:
            # dU = (-1j * dt * H(t+dt/2)).expm()
            # U.append(dU * U[-1])

    # else:
        # U = [u0]

        # for t in tlist[1::]:
            # dU = (-1j * dt * H(t+dt/2)).expm()
            # U[0] = dU * U[0]

    if history:
        U = [u0]
        # creation of tmp necessary to access member function 'evaluate'
        # stupid practice?
        tmp = Qobj()

        dt = tlist[1]

        for t in tlist[1::]:
            h_dt = tmp.evaluate(H, (t+dt/2), args)
            dU = (-1j * dt * h_dt).expm()
            U.append(dU * U[-1])

    else:
        U = [u0]
        # creation of tmp necessary to access member function 'evaluate'
        # stupid practice?
        tmp = Qobj()

        dt = tlist[1]

        for t in tlist[1::]:
            h_dt = tmp.evaluate(H, (t+dt/2), args)
            dU = (-1j * dt * h_dt).expm()
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



