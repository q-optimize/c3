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
    Function for displaying all available devices for tensorflow operations
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

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config = config)


    return sess

def tf_evaluate(tf_obj_list, t, args):
    """
    This function is a 1:1 replication of qutips Qobj.evaluate() function
    but modified for the usecase with tensorflow objects.
    For more information see:
    http://qutip.org/docs/latest/modules/qutip/qobj.html#Qobj.evaluate

    Evaluate a time-dependent tf.Tensor obj in list format. For
    example,

        tf_obj_list = [H0, [H1, func_t]]

    is evaluated to

        tf.Tensor(t) = H0 + H1 * func_t(t, args)

    and

        tf_obj_list = [H0, [H1, 'sin(w * t)']]

    is evaluated to

        tf.Tensor(t) = H0 + H1 * sin(args['w'] * t)

    Parameters
    ----------
    tf_obj_list : list
        A nested list of tf.Tensor obj instances and corresponding time-dependent
        coefficients.
    t : float
        The time for which to evaluate the time-dependent Qobj instance.
    args : dictionary
        A dictionary with parameter values required to evaluate the
        time-dependent tf.Tensor intance.

    Returns
    -------
    output : tf.Tensor
        A tf.Tensor instance that represents the value of tf_obj_list at time t.

    """
    tf_obj_sum = 0
    if isinstance(tf_obj_list, tf.Tensor):
        tf_obj_sum = tf_obj_list
    elif isinstnace(tf_obj_list, list):
        for tf_obj in tf_obj_list:
            if isinstance(tf_obj, tf.Tensor):
                tf_obj_sum += tf_obj
            elif (isinstance(tf_obj, list) and len(tf_obj) == 2 and
                  isinstance(tf_obj[0], tf.Tensor)):
                if isinstance(tf_obj[1], types.FunctionType):
                    tf_obj_sum += tf_obj[0] * tf_obj[1](t, args)
                elif isinstance(tf_obj[1], str):
                    args['t'] = t
                    tf_obj_sum = tf_obj[0] * float(eval(tf_obj[1], globals(), args))
                else:
                    raise TypeError('Unrecognized format for ' +
                                    'specification of time-dependent tf.Tensor')
            else:
                raise TypeError('Unrecognized format for specification ' +
                                'of time-dependent tf.Tensor')
    else:
        raise TypeError(
            'Unrecognized format for specification of time-dependent tf.Tensor')

    return tf_obj_sum


    def matmul_n(tensor_list):
        l = int(tensor_list.shape[0])
        if (l==1):
            return tensor_list[0]
        else:
            even_half = tf.gather(tensor_list, list(range(0,l,2)))
            odd_half = tf.gather(tensor_list, list(range(1,l,2)))
            return tf.matmul(matmul_n(even_half),matmul_n(odd_half))
