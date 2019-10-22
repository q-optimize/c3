"""Various utility functions to speed up tensorflow coding."""

import numpy as np
import tensorflow as tf
from tensorflow.python.client import device_lib
import os


def tf_log_level_info():
    """Display the information about different log levels in tensorflow."""
    info = (
            "Log levels of tensorflow:\n"
            "\t0 = all messages are logged (default behavior)\n"
            "\t1 = INFO messages are not printed\n"
            "\t2 = INFO and WARNING messages are not printed\n"
            "\t3 = INFO, WARNING, and ERROR messages are not printed\n"
            )
    print(info)


def get_tf_log_level():
    """Display the current tensorflow log level of the system."""
    log_lvl = '0'

    if 'TF_CPP_MIN_LOG_LEVEL' in os.environ:
        log_lvl = os.environ['TF_CPP_MIN_LOG_LEVEL']

    return log_lvl


def set_tf_log_level(lvl):
    """
    Set tensorflows system log level.

    REMARK: it seems like the 'TF_CPP_MIN_LOG_LEVEL' variable expects a string.
            the input of this function seems to work with both string and/or
            integer, as casting string to string does nothing. feels hacked?
            but I guess it's just python...
    """
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = str(lvl)


def tf_list_avail_devices():
    """
    List available devices.

    Function for displaying all available devices for tf_setuptensorflow
    operations on the local machine.

    TODO:   Refine output of this function. But without further knowledge
            about what information is needed, best practise is to output all
            information available.
    """
    local_dev = device_lib.list_local_devices()
    print(local_dev)


def tf_setup():
    """Set up the tensorflow environment to be used by c3po."""
    config = tf.ConfigProto(
        allow_soft_placement=True
    )
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    return sess


@tf.function
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
    overlap = tf.linalg.trace(
        tf.matmul(tf.conj(tf.transpose(A)), B)) / tf.cast(B.shape[1], B.dtype)
    return tf.cast(tf.conj(overlap)*overlap, tf.float64)


@tf.function
def tf_measure_operator(M, U):
    return tf.linalg.trace(tf.matmul(M, U))


@tf.function
def tf_expm(A):
    r = tf.eye(int(A.shape[0]), dtype=A.dtype)
    A_powers = A
    r += A

    # TODO dynamically calculate the number of steps required
    for ii in range(2, 14):
        A_powers = tf.matmul(A_powers, A)
        r += A_powers/np.math.factorial(ii)
    return r


@tf.function
def tf_dU_of_t(h0, hks, cflds_t, dt):
    h = h0
    for ii in range(len(hks)):
        h += cflds_t[ii] * hks[ii]
    return tf_expm(-1j*h*dt)


# @tf.function
def tf_dU_of_t_lind(h0, hks, col_ops, cflds_t, dt):
    h = h0
    for ii in range(len(hks)):
        h += cflds_t[ii] * hks[ii]
    lind_op = -1j * (tf_spre(h)-tf_spost(h))
    for col_op in col_ops:
        super_clp = tf.matmul(
                        tf_spre(col_op),
                        tf_spost(tf.linalg.adjoint(col_op))
                        )
        anticomm_L_clp = 0.5 * tf.matmul(
                                    tf_spre(tf.linalg.adjoint(col_op)),
                                    tf_spre(col_op)
                                    )
        anticomm_R_clp = 0.5 * tf.matmul(
                                    tf_spost(col_op),
                                    tf_spost(tf.linalg.adjoint(col_op))
                                    )
        lind_op = lind_op + super_clp - anticomm_L_clp - anticomm_R_clp
    return tf_expm(lind_op * dt)


def tf_propagation(h0, hks, cflds, dt, history=False):
    with tf.name_scope('Propagation'):
        dUs = []
        for ii in range(len(cflds[0])):
            cf_t = []
            for fields in cflds:
                cf_t.append(tf.cast(fields[ii], tf.complex128))
            dUs.append(tf_dU_of_t(h0, hks, cf_t, dt))
        return dUs


def tf_propagation_lind(h0, hks, col_ops, cflds, dt, history=False):
    with tf.name_scope('Propagation'):
        dUs = []
        for ii in range(len(cflds[0])):
            cf_t = []
            for fields in cflds:
                cf_t.append(tf.cast(fields[ii], tf.complex128))
            dUs.append(tf_dU_of_t_lind(h0, hks, col_ops, cf_t, dt))
        return dUs


def tf_matmul_list(dUs):
    U = dUs[0]
    for ii in range(1, len(dUs)):
        U = tf.matmul(dUs[ii], U, name="timestep_"+str(ii))
    return U


def tf_matmul_n(tensor_list):
    l = len(tensor_list)
    if (l == 1):
        return tensor_list[0]
    else:
        left_half = tensor_list[0:int(l/2)]
        right_half = tensor_list[int(l/2):l]
        return tf.matmul(tf_matmul_n(left_half), tf_matmul_n(right_half))


def tf_log10(x):
    """
    Yes, seriously.
    """
    numerator = tf.log(x)
    denominator = tf.log(tf.constant(10, dtype=numerator.dtype))
    return numerator / denominator


def Id_like(A):
    """
    identity of the same size as A
    """
    shape = tf.shape(A)
    dim = shape[0]
    return tf.eye(dim, dtype=tf.complex128)


@tf.function
def tf_spre(A):
    """Superoperator on the left of matrix A."""
    Id = Id_like(A)
    dim = tf.shape(A)[0]
    tensordot = tf.tensordot(A, Id, axes=0)
    reshaped = tf.reshape(
                tf.transpose(tensordot, perm=[0, 2, 1, 3]),
                [dim**2, dim**2]
                )
    return reshaped


@tf.function
def tf_spost(A):
    """Superoperator on the right of matrix A."""
    Id = Id_like(A)
    dim = tf.shape(A)[0]
    tensordot = tf.tensordot(Id, tf.transpose(A), axes=0)
    reshaped = tf.reshape(
                tf.transpose(tensordot, perm=[0, 2, 1, 3]),
                [dim**2, dim**2]
                )
    return reshaped


@tf.function
# TODO needs fixing
def tf_dmdm_fid(rho, sigma):
    rhosqrt = tf.linalg.sqrtm(rho)
    return tf.linalg.trace(
                tf.linalg.sqrtm(
                    tf.matmul(tf.matmul(rhosqrt, sigma), rhosqrt)
                    )
                )


@tf.function
def tf_dmket_fid(rho, psi):
    return tf.sqrt(
            tf.matmul(tf.matmul(tf.linalg.adjoint(psi), rho), psi)
            )


def tf_interp_sin(y_lo, x_lo, x_hi):
    x_left = x_lo[0]
    dx = x_lo[1]-x_lo[0]
    n = int(x_hi.shape[0] / x_lo.shape[0])
    y_hi = []
    for xi in range(1, x_lo.shape[0]):
        x_right = x_lo[xi]
        x = tf.linspace(x_left, x_right, n)
        y = (y_lo[xi-1] + (y_lo[xi] - y_lo[xi-1]) * tf.sin(np.pi/dx*x))
        y_hi = tf.concat(y_hi, y)
        x_left = x_right
    return y_hi
