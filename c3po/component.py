"""Component class and subclasses."""

import types
import numpy as np
import tensorflow as tf
from c3po.utils import num3str
from c3po.hamiltonians import resonator, duffing
from c3po.constants import kb, hbar
from c3po.c3objs import C3obj

class PhysicalComponent(C3obj):
    """
    Represents the components making up a chip.

    Parameters
    ----------
    hilbert_dim: int
        dimension of the Hilbert space representing this physical component

    """

    def __init__(
            self,
            name: str,
            desc: str = " ",
            comment: str = " ",
            hilbert_dim: int = 0,
            ):
        super().__init__(
            name=name,
            desc=desc,
            comment=comment
            )
        self.hilbert_dim = hilbert_dim
        self.Hs = {}
        self.collapse_ops = {}


class Qubit(PhysicalComponent):
    """
    Represents the element in a chip functioning as qubit.

    Parameters
    ----------
    freq: np.float64
        frequency of the qubit
    anhar: np.float64
        anharmonicity of the qubit. defined as w01 - w12
    t1: np.float64
        t1, the time decay of the qubit due to dissipation
    t2star: np.float64
        t2star, the time decay of the qubit due to pure dephasing
    temp: np.float64
        temperature of the qubit, used to determine the Boltzmann distribution
        of energy level populations

    """

    def __init__(
        self,
        name: str,
        desc: str = " ",
        comment: str = " ",
        hilbert_dim: int = 4,
        freq: np.float64 = 0.0,
        anhar: np.float64 = 0.0,
        t1: np.float64 = 0.0,
        t2star: np.float64 = 0.0,
        temp: np.float64 = 0.0
    ):
        super().__init__(
            name=name,
            desc=desc,
            comment=comment,
            hilbert_dim=hilbert_dim
        )
        self.params['freq'] = freq
        if hilbert_dim > 2:
            self.params['anhar'] = anhar
        if t1:
            self.params['t1'] = t1
        if t2star:
            self.params['t2star'] = t2star
        if temp:
            self.params['temp'] = temp

    def init_Hs(self, ann_oper):
        self.Hs['freq'] = tf.constant(
            resonator(ann_oper), dtype=tf.complex128
        )
        if self.hilbert_dim > 2:
            self.Hs['anhar'] = tf.constant(
                duffing(ann_oper), dtype=tf.complex128
            )

    def get_Hamiltonian(self):
        h = tf.cast(
                self.params['freq'].get_value(),
                tf.complex128
        ) * self.Hs['freq']
        if self.hilbert_dim > 2:
            h += tf.cast(
                self.params['anhar'].get_value(),
                tf.complex128
            ) * self.Hs['anhar']
        return h

    def init_Ls(self, ann_oper):
        self.collapse_ops['t1'] = ann_oper
        self.collapse_ops['temp'] = ann_oper.T.conj()
        self.collapse_ops['t2star'] = 2 * tf.matmul(
            ann_oper.T.conj(),
            ann_oper
        )

    def get_Lindbladian(self):
        Ls = []
        if 't1' in self.params:
            t1 = self.params['t1'].get_value()
            gamma = (0.5 / t1) ** 0.5
            L = gamma * self.collapse_ops['t1']
            Ls.append(L)
            if 'temp' in self.params:
                freq_diff = np.array(
                    [(self.params['freq'].get_value()
                      + n*self.params['anhar'].get_value())
                        for n in range(self.hilbert_dim)]
                )
                beta = 1 / (self.params['temp'].get_value() * kb)
                det_bal = tf.exp(-hbar*freq_diff*beta)
                det_bal_mat = tf.linalg.tensor_diag(det_bal)
                L = gamma * tf.matmul(self.collapse_ops['temp'], det_bal_mat)
                Ls.append(L)
        if 't2star' in self.params:
            gamma = (0.5/self.params['t2star'].get_value())**0.5
            L = gamma * self.collapse_ops['t2star']
            Ls.append(L)
        return tf.cast(sum(Ls), tf.complex128)


class Resonator(PhysicalComponent):
    """
    Represents the element in a chip functioning as resonator.

    Parameters
    ----------
    freq: np.float64
        frequency of the resonator

    """

    def __init__(
            self,
            name: str,
            desc: str = " ",
            comment: str = " ",
            hilbert_dim: int = 4,
            freq: np.float64 = 0.0
            ):
        super().__init__(
            name=name,
            desc=desc,
            comment=comment,
            hilbert_dim=hilbert_dim
            )
        self.params['freq'] = freq

    def init_Hs(self, ann_oper):
        self.Hs['freq'] = tf.constant(
            resonator(ann_oper), dtype=tf.complex128
        )

    def get_Hamiltonian(self):
        freq = tf.cast(self.params['freq'].get_value(), tf.complex128)
        return freq * self.Hs['freq']


class LineComponent(C3obj):
    """
    Represents the components connecting chip elements and drives.

    Parameters
    ----------
    connected: list
        specifies the component that are connected with this line

    """

    def __init__(
            self,
            name: str,
            desc: str = " ",
            comment: str = " ",
            connected: list = [],
            hamiltonian_func: types.FunctionType = None,
            ):
        super().__init__(
            name=name,
            desc=desc,
            comment=comment
            )
        self.connected = connected
        self.Hs = {}


class Coupling(LineComponent):
    """
    Represents a coupling behaviour between elements.

    Parameters
    ----------
    strength: np.float64
        coupling strength
    connected: list
        all physical components coupled via this specific coupling

    """

    def __init__(
            self,
            name: str,
            desc: str = " ",
            comment: str = " ",
            connected: list = [],
            strength: np.float64 = 0.0,
            hamiltonian_func: types.FunctionType = None,
            ):
        super().__init__(
            name=name,
            desc=desc,
            comment=comment,
            connected=connected,
            hamiltonian_func=hamiltonian_func
            )
        self.hamiltonian_func = hamiltonian_func
        self.params['strength'] = strength

    def init_Hs(self, ann_opers):
        opers_list = [
            ann_opers[conn_comp] for conn_comp in self.connected
        ]
        self.Hs['strength'] = tf.constant(
            self.hamiltonian_func(opers_list), dtype=tf.complex128
        )

    def get_Hamiltonian(self):
        return self.params['strength'] * self.Hs['strength']


class Drive(LineComponent):
    """
    Represents a drive line.

    Parameters
    ----------
    connected: list
        all physical components recieving driving signals via this line

    """

    def __init__(
            self,
            name: str,
            desc: str = " ",
            comment: str = " ",
            connected: list = [],
            hamiltonian_func: types.FunctionType = None,
            ):
        super().__init__(
            name=name,
            desc=desc,
            comment=comment,
            connected=connected,
            hamiltonian_func=hamiltonian_func
            )
        self.hamiltonian_func = hamiltonian_func

    def init_Hs(self, ann_opers: dict):
        hs = []
        for key in self.connected:
            hs.append(
                tf.constant(
                    self.hamiltonian_func(ann_opers[key]), dtype=tf.complex128
                )
            )
        self.h = sum(hs)

    def get_Hamiltonian(self):
        return self.h
