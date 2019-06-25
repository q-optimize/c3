
# REMARK: Until a clearer specification of the way the hamiltonian is passed
#         is given, i.e. what is the exact return value/format of
#         get_Hamiltonian of class model this part stays quiet vague.

#
# !!! CAUTION !!!
# None of this code has been tested yet!

import matplotlib.pyplot as plt


from qutip import Qobj
from qutip import sesolve

from c3po.utils import tf_utils

import tensorflow as tf
import numpy as np
import scipy as sp


def conv_func(tf_sess, func):
    return func


def dirty_wrap(tf_sess, func):
    f = conv_func(tf_sess, func)
    return lambda t, args: f(t)


# def propagate(model, gate, u0, tlist, method, tf_sess = None, grad = False, history = False):
    # """
    # Wrapper function for choosing the type of propagation method.
    # :param model:   :class:'c3po.model'
       # Meta class for system hamiltonian
    # :param u0:
       # can be a state, a unitary or ...
    # :param tlist:
       # ...
    # :param  method:
       # switch for choosing the desired propagator. options are: "pwc",
       # "pwc_tf", "qutip_sesolv"

    # """
    # methods = ["pwc", "qutip_sesolve"]

    # if method not in methods:
        # raise Exception('Method is not supported!')


    # # this is garbage and needs to be fixed. either get_control_fields
    # # delivers a list of all control fields or only a function
    # # that is added here to a list
    # keys = gate.get_parameters().keys()

    # control_fields = []
    # for key in keys:
        # cflds = gate.get_control_fields(key)
        # for cf in cflds:
            # control_fields.append(cf)


    # # dictrionary of parameters for control fields
    # # but this is kind of obsolete as sesolve won't need arguments(?)
    # params = gate.get_parameters()  # retrieve parameters/args for the drive
                                    # # fields

    # params.update(model.get_params_tf())    # system parameters should be provided
                                            # # by model for the tensorflow backend
                                            # # as input for:
                                            # # session.run(..., feed_dict = params)

    # u0_real = tf.convert_to_tensor(u0.full().real, dtype=tf.float32)
    # u0_imag = tf.convert_to_tensor(u0.full().imag, dtype=tf.float32)
    # u0_tf = tf.complex(u0_real, u0_imag)

    # params.update(u0_tf) # u0 should be part of the params passed to
                          # # tensorflow and also needs to be a initialized/converted
                        # # as tensorflow object

    # # should return Hamilton as list [H0, ...]
    # hlist = model.get_tf_Hamiltonian(control_fields)


    # if method == "pwc":
        # U = sesolve_pwc(hlist, u0, tlist, params, tf_sess, grad, history="True")

    # return U


def sesolve_pwc(hlist, u0, tlist, tf_sess, grad = False, history = False):
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


    H = []
    for i in range(0, len(hlist)):
        if i == 0:
            h0 = tf_sess.run(hlist[i])
            tmp = Qobj(h0)
            H.append(tmp)
        else:
            hd = tf_sess.run(hlist[i][0])
            tmp = Qobj(hd)
            H.append(hd)


    t_start = tlist[0]
    t_final = tlist[len(tlist) - 1]
    N_slices = len(tlist)

    Ts = tf.linspace(t_start, t_final, N_slices, name="Time")
    Ts = tf.cast(Ts, tf.float64)
    dt_tf = Ts[1]

    clist = []
    for h in hlist:
        if isinstance(h, list):
            clist.append(tf_sess.run(h[1]))

    H_eval_t = []
    for i in range(0, len(tlist)):
        hdt = H[0]
        for j in range(1, len(H)):
            hdt += clist[j - 1][i] * H[j]
        H_eval_t.append(hdt)

    dt = tlist[1]
    print(H_eval_t)
    if history:
        U = [u0]
        # creation of tmp necessary to access member function 'evaluate'
        # stupid practice?


        for i in range(0, len(tlist)):
            dU = (-1j * dt * H_eval_t[i]).expm()
            dU.dims = U[-1].dims
            U.append(dU * U[-1])

    else:
        U = [u0]
        # creation of tmp necessary to access member function 'evaluate'
        # stupid practice?

        for i in range(0, len(tlist)):
            dU = (-1j * dt * H_eval_t[i]).expm()
            U[0] = dU * U[0]

    return U


def sesolve_pwc_tf(hlist, u0, tlist, tf_sess, history = False):
    """
    Find the propagator of a system Hamiltonian H(t). The initial basis u0. The
    hamiltonian
    :param H:  given in tensorflow
        System hamiltonian or a list of Drift and Control terms
    """

    H_t_eval = []


    t_placeholder = tf.placeholder(tf.float64)

    hdt = hlist[0]
    for i in range(1, len(hlist)):
        hdrive = hlist[i][0]
        cf = hlist[i][1]
        hdt += tf.cast(cf(t_placeholder), tf.complex128) * hdrive

    for i in range(0, len(tlist)):
        H_t_eval.append(tf_sess.run(hdt, feed_dict={t_placeholder: tlist[i]}))



    dt_placeholder = tf.constant(tlist[1], tf.complex128)
    u_old = tf.placeholder(tf.complex128)
    hbar = tf.constant(1, dtype=tf.complex128, name='planck')
    unit_img = tf.constant(-1j, dtype=tf.complex128)
    expm_h = tf.linalg.expm(unit_img * hdt * dt_placeholder /hbar)
    u_new =  tf.linalg.matmul(expm_h, u_old)


    dt = tlist[1]
    ulist = []
    u_initial = u0.full()
    ulist.append(u_initial)
    for i in range(0, len(tlist)):
        if i == 0:
            u_old_input = u_initial
        else:
            u_old_input = u_new_eval
        u_new_eval = tf_sess.run(u_new, feed_dict={t_placeholder: (tlist[i] + dt/2.0), u_old: u_old_input})
        ulist.append(u_new_eval)

    return ulist
