import tensorflow as tf

def resonator(a):
    """
    Builds a harmonic oscillator from the given annihilator. All Hamiltonian
    components are designed to be multiplied with exactly one model parameter.

    Parameters
    ----------
    a : Tensor
        Annihilator.

    Returns
    -------
    Tensor
        Number operator.

    """
    a_dag = tf.transpose(tf.conj(a))
    return tf.matmul(a_dag, a)


def duffing(a):
    """
    Anharmonic part of the duffing oscillator. All Hamiltonian components are
    designed to be multiplied with exactly one model parameter.

    Parameters
    ----------
    a : Tensor
        Annihilator.

    Returns
    -------
    Tensor
        Number operator.

    """
    a_dag = tf.transpose(tf.conj(a))
    n = tf.matmul(a_dag, a)
    return 1/2 * tf.matmul(n - 1, n)


def int_XX(anhs):
    """
    Dipole type coupling. All Hamiltonian components are designed to be
    multiplied with exactly one model parameter.

    Parameters
    ----------
    anhs : Tensor list
        Annihilators.

    Returns
    -------
    Tensor
        coupling

    """
    a = anhs[0]
    b = anhs[1]
    a_dag = tf.transpose(tf.conj(a))
    b_dag = tf.transpose(tf.conj(b))
    return tf.matmul(a_dag + a, b_dag + b)


def drive(anhs):
    """
    Semiclassical drive. All Hamiltonian components are designed to be
    multiplied with exactly one model parameter.

    Parameters
    ----------
    a : Tensor
        Annihilator.

    Returns
    -------
    Tensor
        Number operator.

    """
    a = anhs[0]
    a_dag = tf.transpose(tf.conj(a))
    return a_dag + a
