import tensorflow as tf
from tensorflow.python.client import device_lib
import tensorflow as tf
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
        intra_op_parallelism_threads=0,
        inter_op_parallelism_threads=0,
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
        even_half = tf.gather(tensor_list, list(range(0,l,2)))
        odd_half = tf.gather(tensor_list, list(range(1,l,2)))
        return tf.matmul(matmul_n(even_half),matmul_n(odd_half))


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
        tf.matmul(tf.transpose(A), B)) / tf.cast(B.shape[1], B.dtype)
        )


def tf_dU_of_t(h0, hks, cflds_t, dt):
    h = h0
    for ii in range(len(hks)):
            h += cflds_t[ii]*hks[ii]

    return tf.linalg.expm(-1j*h*dt)


def tf_propagation(h0, hks, cflds, dt, history=False):
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
            U = tf.matmul(tf.gather(dUs, ii), U)

        return U
