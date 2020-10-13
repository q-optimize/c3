"""Component class and subclasses for the components making up the quantum device."""

import types
import numpy as np
import tensorflow as tf
from c3po.libraries.hamiltonians import resonator, duffing
from c3po.libraries.constants import kb, hbar
from c3po.utils.tf_utils import tf_diff
from c3po.utils.qt_utils import hilbert_space_kron as hskron
from c3po.c3objs import C3obj


class PhysicalComponent(C3obj):
    """
    Represents the components making up a chip.

    Parameters
    ----------
    hilbert_dim : int
        Dimension of the Hilbert space of this component

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

    def set_subspace_index(self, index):
        self.index = index


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
        """
        Initialize the qubit Hamiltonians. If the dimension is higher than two, a Duffing oscillator is used.

        Parameters
        ----------
        ann_oper : np.array
            Annihilation operator in the full Hilbert space

        """
        self.Hs['freq'] = tf.constant(
            resonator(ann_oper), dtype=tf.complex128
        )
        if self.hilbert_dim > 2:
            self.Hs['anhar'] = tf.constant(
                duffing(ann_oper), dtype=tf.complex128
            )

    def get_Hamiltonian(self):
        """
        Compute the Hamiltonian. Multiplies the number operator with the frequency and anharmonicity with
        the Duffing part and returns their sum.

        Returns
        -------
        tf.Tensor
            Hamiltonian

        """
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
        """
        Initialize Lindbladian components.

        Parameters
        ----------
        ann_oper : np.array
            Annihilation operator in the full Hilbert space

        """
        self.collapse_ops['t1'] = ann_oper
        self.collapse_ops['temp'] = ann_oper.T.conj()
        self.collapse_ops['t2star'] = 2 * tf.matmul(
            ann_oper.T.conj(),
            ann_oper
        )

    def get_Lindbladian(self, dims):
        """
        Compute the Lindbladian, based on relaxation, dephasing constants and finite temperature.

        Returns
        -------
        tf.Tensor
            Hamiltonian

        """
        Ls = []
        if 't1' in self.params:
            t1 = self.params['t1'].get_value()
            gamma = (0.5 / t1) ** 0.5
            L = gamma * self.collapse_ops['t1']
            Ls.append(L)
            if 'temp' in self.params:
                if self.hilbert_dim > 2:
                    freq_diff = np.array(
                        [(self.params['freq'].get_value()
                          + n*self.params['anhar'].get_value())
                            for n in range(self.hilbert_dim)]
                    )
                else:
                    freq_diff = np.array(
                        [self.params['freq'].get_value(), 0]
                    )
                beta = 1 / (self.params['temp'].get_value() * kb)
                det_bal = tf.exp(-hbar*tf.cast(freq_diff, tf.float64)*beta)
                det_bal_mat = hskron(
                    tf.linalg.tensor_diag(det_bal), self.index, dims
                )
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
        """
        Initialize the Hamiltonian as a number operator

        Parameters
        ----------
        ann_oper : np.array
            Annihilation operator in the full Hilbert space.

        """
        self.Hs['freq'] = tf.constant(
            resonator(ann_oper), dtype=tf.complex128
        )

    def init_Ls(self, ann_oper):
        """NOT IMPLEMENTED"""
        pass

    def get_Hamiltonian(self):
        """Compute the Hamiltonian."""
        freq = tf.cast(self.params['freq'].get_value(), tf.complex128)
        return freq * self.Hs['freq']

    def get_Lindbladian(self, dims):
        """NOT IMPLEMENTED"""
        pass


class SymmetricTransmon(PhysicalComponent):
    """
    Represents the element in a chip functioning as tunanble coupler.

    Parameters
    ----------
    freq: np.float64
        base frequency of the TC
    phi_0: np.float64
        half period of the phase dependant function
    phi: np.fl

    """

    def __init__(
            self,
            name: str,
            desc: str = " ",
            comment: str = " ",
            hilbert_dim: int = 2,
            freq: np.float64 = 0.0,
            phi: np.float64 = 0.0,
            phi_0: np.float64 = 0.0
            ):
        super().__init__(
            name=name,
            desc=desc,
            comment=comment,
            hilbert_dim=hilbert_dim
            )
        self.params['freq'] = freq
        self.params['phi'] = phi
        self.params['phi_0'] = phi_0

    def init_Hs(self, ann_oper):
        self.Hs['freq'] = tf.constant(
            resonator(ann_oper), dtype=tf.complex128
        )

    def init_Ls(self, ann_oper):
        pass

    def get_Hamiltonian(self):
        freq = tf.cast(self.params['freq'].get_value(), tf.complex128)
        pi = tf.constant(np.pi, dtype=tf.complex128)
        phi = tf.cast(self.params['phi'].get_value(), tf.complex128)
        phi_0 = tf.cast(self.params['phi_0'].get_value(), tf.complex128)
        return freq * tf.cast(tf.sqrt(tf.abs(tf.cos(
            pi * phi / phi_0
        ))), tf.complex128) * self.Hs['freq']



class AsymmetricTransmon(PhysicalComponent):
    """
    Represents the element in a chip functioning as tunanble coupler.

    Parameters
    ----------
    freq: np.float64
        base frequency of the TC
    phi_0: np.float64
        half period of the phase dependant function
    phi: np.fl

    """

    def __init__(
            self,
            name: str,
            desc: str = " ",
            comment: str = " ",
            hilbert_dim: int = 2,
            freq: np.float64 = 0.0,
            phi: np.float64 = 0.0,
            phi_0: np.float64 = 0.0,
#             gamma: np.float64 = 0.0,
            d: np.float64 = 0.0,
            ):
        super().__init__(
            name=name,
            desc=desc,
            comment=comment,
            hilbert_dim=hilbert_dim
            )
        self.params['freq'] = freq
        self.params['phi'] = phi
        self.params['phi_0'] = phi_0
        self.params['d'] = d
#         self.params['gamma'] = gamma

    def init_Hs(self, ann_oper):
        self.Hs['freq'] = tf.constant(
            resonator(ann_oper), dtype=tf.complex128
        )

    def init_Ls(self, ann_oper):
        pass

    def get_Hamiltonian(self):
        freq = tf.cast(self.params['freq'].get_value(), tf.complex128)
        pi = tf.constant(np.pi, dtype=tf.complex128)
        phi = tf.cast(self.params['phi'].get_value(), tf.complex128)
        phi_0 = tf.cast(self.params['phi_0'].get_value(), tf.complex128)
        d = tf.cast(self.params['d'].get_value(), tf.complex128)
#         gamma = tf.cast(self.params['gamma'].get_value(), tf.complex128)
#         d = (gamma - 1) / (gamma + 1)
        factor = tf.sqrt(tf.sqrt(
            tf.cos(pi * phi / phi_0)**2 + d**2 * tf.sin(pi * phi / phi_0)**2
        ))
        return freq * factor * self.Hs['freq']

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

    def init_Hs(self, opers_list):
        self.Hs['strength'] = tf.constant(
            self.hamiltonian_func(opers_list), dtype=tf.complex128
        )

    def get_Hamiltonian(self):
        strength = tf.cast(self.params['strength'].get_value(), tf.complex128)
        return strength * self.Hs['strength']


class Drive(LineComponent):
    """
    Represents a drive line.

    Parameters
    ----------
    connected: list
        all physical components receiving driving signals via this line

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

    def init_Hs(self, ann_opers: list):
        hs = []
        for a in ann_opers:
            hs.append(
                tf.constant(
                    self.hamiltonian_func(a), dtype=tf.complex128
                )
            )
        self.h = sum(hs)

    def get_Hamiltonian(self):
        return self.h
