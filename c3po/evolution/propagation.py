
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


# def sesolve_pwc(H, u0, tlist, args={}, tf_sess, grad = False, history = False):
    # """
    # Find the propagator of a system Hamiltonian H(t). The initial basis u0. The
    # hamiltonian
    # :param H:   :class:'qutip.qobj'
        # System hamiltonian or a list of Drift and Control terms
    # :param u0:
        # can be a state, a unitary or a rectangular matrix made of several
        # initial states.
    # :param tlist:
        # time vector
    # :output:    list of unitaries (Qobj)
        # for all times in tlist or just the initial and final time (depending
        # on history setting)
    # """


# # CODE TAKEN FROM FEDERICO'S CODE: new_IBM_USAAR/utils.py

# #     if history:
        # # U = [u0]

        # # for t in tlist[1::]:
            # # dU = (-1j * dt * H(t+dt/2)).expm()
            # # U.append(dU * U[-1])

    # # else:
        # # U = [u0]

        # # for t in tlist[1::]:
            # # dU = (-1j * dt * H(t+dt/2)).expm()
            # # U[0] = dU * U[0]

    # if history:
        # U = [u0]
        # # creation of tmp necessary to access member function 'evaluate'
        # # stupid practice?
        # tmp = Qobj()

        # dt = tlist[1]

        # for t in tlist[1::]:
            # h_dt = tmp.evaluate(H, (t+dt/2), args)
            # dU = (-1j * dt * h_dt).expm()
            # U.append(dU * U[-1])

    # else:
        # U = [u0]
        # # creation of tmp necessary to access member function 'evaluate'
        # # stupid practice?
        # tmp = Qobj()

        # dt = tlist[1]

        # for t in tlist[1::]:
            # h_dt = tmp.evaluate(H, (t+dt/2), args)
            # dU = (-1j * dt * h_dt).expm()
            # U[0] = dU * U[0]



    # return U


def sesolve_pwc_tf(hlist, u0, tlist, tf_sess, history = False):
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


    u_initial = tf.constant(u0.full(), dtype=tf.complex128, name="u_initial")

    t_start = tlist[0]
    t_final = tlist[len(tlist) - 1]
    N_slices = len(tlist)
#    N_slices = 10

    print("N_slices:" + str(N_slices))

    Ts = tf.linspace(t_start, t_final, N_slices, name="Time")
    Ts = tf.cast(Ts, tf.float64)
    dt = Ts[1]


    clist = []
    for h in hlist:
        if isinstance(h, list):
            print(h)
            c = h[1](tf.math.add(Ts, dt/2))
            clist.append(c)

#    c = hlist[1][1](tf.math.add(Ts, dt/2))

#    cl = tf_sess.run(clist[0])

    H_t = []
    H_t_eval = []
    for i in range(0, N_slices):
        h_dt = 0
        for j in range(0, len(hlist)):
            if j == 0:
                h_dt = hlist[j]
            else:
                h_dt += tf.cast(clist[j - 1][i], tf.complex128) * hlist[j][0]
        H_t.append(h_dt)
        H_t_eval.append(tf_sess.run(h_dt))

#     with tf.name_scope('U_actual'):
        # def condition(i, u):
            # return i < N_slices

        # def body(i, u):
            # hbar = tf.constant(1, dtype=tf.complex128, name='planck')
            # hdt = tf.gather(H_t, i)
            # u_ = tf.linalg.expm(- tf.constant(1j, dtype=tf.complex128) / hbar * (hdt * tf.cast(dt, tf.complex128))) * u
            # return i+1, u_

# #        max_i, uf = tf.while_loop(condition, body, (0, u_initial), parallel_interactions=10)
        # max_i, uf = tf.while_loop(condition, body, (0, u_initial))

    ulist = []
#     hbar = tf.constant(1, dtype=tf.complex128, name='planck')
    # ulist.append(tf_sess.run(u_initial))
    # for i in range(N_slices):
        # print("step: " + str(i))
        # if i == 0 :
            # u_old = u_initial
        # else:
            # u_old = tf.constant(u_new_eval, dtype=tf.complex128)
# #        hdt = tf.gather(H_t, i)
        # hdt = H_t_eval[i]
        # hdt = tf.constant(hdt, dtype=tf.complex128)
        # u_new = tf.linalg.expm(- tf.constant(1j, dtype=tf.complex128)/ hbar * (hdt * tf.cast(dt, tf.complex128))) * u_old
        # u_new_eval = tf_sess.run(u_new)
# #        print(u_new_eval)
        # ulist.append(u_new_eval)

    u_initial = np.matrix(u0.full())
    dt = tlist[1]

    hbar = 1
    for i in range(N_slices):
        print("step: " + str(i))
        if i == 0 :
            u_old = u_initial
        else:
            u_old = u_new
        hdt = np.matrix(H_t_eval[i])
#         print('hdt')
        # print(hdt)
        # print(' ')
        # print('dt')
        # print(dt)
        # print(' ')
        # print('u_old')
        # print(u_old)
        u_new = sp.linalg.expm(-1j / hbar * hdt * dt) * u_old
        ulist.append(u_new.A)





#     with session.as_default():
        # run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
        # run_metadata = tf.RunMetadata()
        # session.run(H, feed_dict=params)

#    out = tf_sess.run(uf)


    return ulist



