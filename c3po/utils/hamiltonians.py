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
    a_dag = tf.linalg.adjoint(a)
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
    a_dag = tf.linalg.adjoint(a)
    n = tf.matmul(a_dag, a)
    return 1/2 * tf.matmul(n - tf.eye(int(n.shape[0]), dtype=tf.complex128), n)


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
    a_dag = tf.linalg.adjoint(a)
    b_dag = tf.linalg.adjoint(b)
    return tf.matmul(a_dag + a, b_dag + b)


def x_drive(anhs):
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
    anhs_dag = tf.linalg.adjoint(anhs)
    return anhs_dag + anhs
