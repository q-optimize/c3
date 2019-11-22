"""Library of Hamiltonian functions."""
import numpy as np


def resonator(a):
    """
    Harmonic oscillator hamiltonian given the annihilation operator.

    Parameters
    ----------
    a : Tensor
        Annihilator.

    Returns
    -------
    Tensor
        Number operator.

    """
    a_dag = a.T.conj()
    return np.matmul(a_dag, a)


def duffing(a):
    """
    Anharmonic part of the duffing oscillator.

    Parameters
    ----------
    a : Tensor
        Annihilator.

    Returns
    -------
    Tensor
        Number operator.

    """
    a_dag = a.T.conj()
    n = np.matmul(a_dag, a)
    return 1/2 * np.matmul(n - np.eye(int(n.shape[0]), dtype=np.complex128), n)


def int_XX(anhs):
    """
    Dipole type coupling.

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
    a_dag = a.T.conj()
    b_dag = b.T.conj()
    return np.matmul(a_dag + a, b_dag + b)


def int_YY(anhs):
    """
    Dipole type coupling.

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
    a_dag = a.T.conj()
    b_dag = b.T.conj()
    return -np.matmul(a_dag - a, b_dag - b)


def x_drive(a):
    """
    Semiclassical drive.

    Parameters
    ----------
    a : Tensor
        Annihilator.

    Returns
    -------
    Tensor
        Number operator.

    """
    a_dag = a.T.conj()
    return a_dag + a


def y_drive(a):
    """
    Semiclassical drive.

    Parameters
    ----------
    a : Tensor
        Annihilator.

    Returns
    -------
    Tensor
        Number operator.

    """
    a_dag = a.T.conj()
    return 1.0j * (a_dag - a)
