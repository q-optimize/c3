import numpy as np
import tensorflow as tf
from tensorflow.python.client import device_lib
import os
import types


def tf_log_level_info():
    """
    Function for displaying the information about different log levels in
    tensorflow.
    """

    info =  (
            "Log levels of tensorflow:\n"
            "\t0 = all messages are logged (default behavior)\n"
            "\t1 = INFO messages are not printed\n"
            "\t2 = INFO and WARNING messages are not printed\n"
            "\t3 = INFO, WARNING, and ERROR messages are not printed\n"
            )

    print(info)


def get_tf_log_level():
    """
    Function for displaying the current tensorflow log level of the system.
    """
    log_lvl = '0'

    if 'TF_CPP_MIN_LOG_LEVEL' in os.environ:
        log_lvl = os.environ['TF_CPP_MIN_LOG_LEVEL']

    return log_lvl


def set_tf_log_level(lvl):
    """
    Function for setting tensorflows system log level.

    REMARK: it seems like the 'TF_CPP_MIN_LOG_LEVEL' variable expects a string.
            the input of this function seems to work with both string and/or
            integer, as casting string to string does nothing. feels hacked?
            but I guess it's just python...
    """
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = str(lvl)



def tf_list_avail_devices():
    """
    Function for displaying all available devices for tf_setuptensorflow operations
    on the local machine.

    TODO:   Refine output of this function. But without further knowledge
            about what information is needed, best practise is to output all
            information available.
    """
    local_dev = device_lib.list_local_devices()
    print(local_dev)


def tf_setup():
    """
    Function for setting up the tensorflow environment to be used by c3po
    """
    config = tf.ConfigProto(
        allow_soft_placement=True
    )
    config.gpu_options.allow_growth = True
    sess = tf.Session(config = config)
    return sess


def tf_matmul_n(tensor_list):
    l = int(tensor_list.shape[0])
    if (l==1):
        return tensor_list[0]
    else:
        left_half = tf.gather(tensor_list, list(range(0,int(l/2))))
        right_half = tf.gather(tensor_list, list(range(int(l/2),l)))
        return tf.matmul(tf_matmul_n(left_half), tf_matmul_n(right_half))


def tf_unitary_overlap(A, B):
    """
    Unitary overlap between two matrices in Tensorflow(tm).

    Parameters
    ----------
    A : Tensor
        Description of parameter `A`.
    B : Tensor
        Description of parameter `B`.

    Returns
    -------
    type
        Description of returned object.
    """
    return tf.abs(tf.linalg.trace(
        tf.matmul(tf.transpose(A), B))/ tf.cast(B.shape[1], B.dtype),
        name = "unitary_overlap"
        )

def tf_measure_operator(M, U):
    return tf.linalg.trace(tf.matmul(M, U))


def tf_expm(A):
    r = tf.eye(int(A.shape[0]), dtype=A.dtype)
    A_powers = A
    r += A

    for ii in range(2,8):
        A_powers = tf.matmul(A_powers, A)
        r += A_powers/np.math.factorial(ii)

    return r


def tf_dU_of_t(h0, hks, cflds_t, dt):
    h = h0
    for ii in range(len(hks)):
            h += cflds_t[ii]*hks[ii]

    return tf.linalg.expm(-1j*h*dt)


def tf_propagation(h0, hks, cflds, dt, history=False):
    with tf.name_scope('Propagation'):
        control_fields = tf.cast(
            tf.transpose(tf.stack(cflds)),
            tf.complex128,
            name='Control_fields'
            )

        dUs = tf.map_fn(
            lambda fields: tf_dU_of_t(h0, hks, fields, dt),
            control_fields,
            name='dU_of_t'
            )

        if history:
            u_t = tf.gather(dUs,0)
            history = [u_t]
            for ii in range(dUs.shape[0]-1):
                du = tf.gather(dUs, ii+1)
                u_t = tf.matmul(du,u_t)
                history.append(u_t)
            return history, ts
        else:
            U = tf.gather(dUs, 0)
            for ii in range(1, dUs.shape[0]):
                U = tf.matmul(tf.gather(dUs, ii), U, name="timestep_"+str(ii))

        return U

def tf_log10(x):
    """
    Yes, seriously.
    """
    numerator = tf.log(x)
    denominator = tf.log(tf.constant(10, dtype=numerator.dtype))
    return numerator / denominator
