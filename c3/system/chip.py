"""Component class and subclasses for the components making up the quantum device."""

import numpy as np
import tensorflow as tf

from c3.c3objs import C3obj
from c3.libraries.constants import kb, hbar
from c3.libraries.hamiltonians import hamiltonians
from c3.utils.qt_utils import hilbert_space_kron as hskron


device_lib = dict()


def dev_reg_deco(func):
    """
    Decorator for making registry of functions
    """
    device_lib[str(func.__name__)] = func
    return func


class PhysicalComponent(C3obj):
    """
    Represents the components making up a chip.

    Parameters
    ----------
    hilbert_dim : int
        Dimension of the Hilbert space of this component

    """

    def __init__(self, **props):
        self.params = {}
        self.hilbert_dim = props.pop("hilbert_dim", None)
        super().__init__(**props)
        self.Hs = {}
        self.collapse_ops = {}
        self.drive_line = None

    def set_subspace_index(self, index):
        self.index = index

    def asdict(self) -> dict:
        params = {}
        for key, item in self.params.items():
            params[key] = item.asdict()
        return {
            "c3type": self.__class__.__name__,
            "params": params,
            "hilbert_dim": self.hilbert_dim,
        }


@dev_reg_deco
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
        name,
        hilbert_dim,
        desc=None,
        comment=None,
        freq=None,
        anhar=None,
        t1=None,
        t2star=None,
        temp=None,
        params=None,
    ):
        # TODO Cleanup params passing and check for conflicting information
        super().__init__(
            name=name,
            desc=desc,
            comment=comment,
            hilbert_dim=hilbert_dim,
            params=params,
        )
        if freq:
            self.params['freq'] = freq
        if anhar:
            self.params['anhar'] = anhar
        if t1:
            self.params["t1"] = t1
        if t2star:
            self.params["t2star"] = t2star
        if temp:
            self.params["temp"] = temp

    def init_Hs(self, ann_oper):
        """
        Initialize the qubit Hamiltonians. If the dimension is higher than two, a
        Duffing oscillator is used.

        Parameters
        ----------
        ann_oper : np.array
            Annihilation operator in the full Hilbert space

        """
        resonator = hamiltonians["resonator"]
        self.Hs["freq"] = tf.Variable(resonator(ann_oper), dtype=tf.complex128)
        if self.hilbert_dim > 2:
            duffing = hamiltonians["duffing"]
            self.Hs["anhar"] = tf.Variable(duffing(ann_oper), dtype=tf.complex128)

    def get_Hamiltonian(self):
        """
        Compute the Hamiltonian. Multiplies the number operator with the frequency and
        anharmonicity with the Duffing part and returns their sum.

        Returns
        -------
        tf.Tensor
            Hamiltonian

        """
        h = tf.cast(self.params["freq"].get_value(), tf.complex128) * self.Hs["freq"]
        if self.hilbert_dim > 2:
            anhar = tf.cast(self.params["anhar"].get_value(), tf.complex128)
            h += anhar * self.Hs["anhar"]
        return h 
    
    
    def init_Ls(self, ann_oper):
        """
        Initialize Lindbladian components.

        Parameters
        ----------
        ann_oper : np.array
            Annihilation operator in the full Hilbert space

        """
        self.collapse_ops["t1"] = ann_oper
        self.collapse_ops["temp"] = ann_oper.T.conj()
        self.collapse_ops["t2star"] = 2 * tf.matmul(ann_oper.T.conj(), ann_oper)

    def get_Lindbladian(self, dims):
        """
        Compute the Lindbladian, based on relaxation, dephasing constants and finite
        temperature.

        Returns
        -------
        tf.Tensor
            Hamiltonian

        """
        Ls = []
        if "t1" in self.params:
            t1 = self.params["t1"].get_value()
            gamma = (0.5 / t1) ** 0.5
            L = gamma * self.collapse_ops["t1"]
            Ls.append(L)
            if "temp" in self.params:
                if self.hilbert_dim > 2:
                    freq = self.params["freq"].get_value()
                    anhar = self.params["anhar"].get_value()
                    freq_diff = np.array(
                        [freq + n * anhar for n in range(self.hilbert_dim)]
                    )
                else:
                    freq_diff = np.array([self.params["freq"].get_value(), 0])
                beta = 1 / (self.params["temp"].get_value() * kb)
                det_bal = tf.exp(-hbar * tf.cast(freq_diff, tf.float64) * beta)
                det_bal_mat = hskron(tf.linalg.tensor_diag(det_bal), self.index, dims)
                L = gamma * tf.matmul(self.collapse_ops["temp"], det_bal_mat)
                Ls.append(L)
        if "t2star" in self.params:
            gamma = (0.5 / self.params["t2star"].get_value()) ** 0.5
            L = gamma * self.collapse_ops["t2star"]
            Ls.append(L)
        if Ls == []:
            raise Exception("No T1 or T2 provided")
        return tf.cast(sum(Ls), tf.complex128)
    

@dev_reg_deco
class Resonator(PhysicalComponent):
    """
    Represents the element in a chip functioning as resonator.

    Parameters
    ----------
    freq: np.float64
        frequency of the resonator

    """

    def init_Hs(self, ann_oper):
        """
        Initialize the Hamiltonian as a number operator

        Parameters
        ----------
        ann_oper : np.array
            Annihilation operator in the full Hilbert space.

        """
        self.Hs["freq"] = tf.Variable(
            hamiltonians["resonator"](ann_oper), dtype=tf.complex128
        )

    def init_Ls(self, ann_oper):
        """NOT IMPLEMENTED"""
        pass

    def get_Hamiltonian(self):
        """Compute the Hamiltonian."""
        freq = tf.cast(self.params["freq"].get_value(), tf.complex128)
        return freq * self.Hs["freq"]

    def get_Lindbladian(self, dims):
        """NOT IMPLEMENTED"""
        pass


@dev_reg_deco
class Transmon(PhysicalComponent):
    """
    Represents the element in a chip functioning as tunanble transmon qubit.

    Parameters
    ----------
    freq: np.float64
        base frequency of the Transmon
    phi_0: np.float64
        half period of the phase dependant function
    phi: np.float64
        flux position

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
        gamma: np.float64 = None,
        d: np.float64 = None,
        t1: np.float64 = 0.0,
        t2star: np.float64 = 0.0,
        temp: np.float64 = 0.0,
        anhar: np.float64 = 0.0,
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

        if d:
            self.params['d'] = d
        elif gamma:
            self.params['gamma'] = gamma
        else:
            raise Warning(
                "no gamma or d provided. setting d=0, i.e. symmetric case"
            )
        if hilbert_dim > 2:
            self.params['anhar'] = anhar
        if t1:
            self.params['t1'] = t1
        if t2star:
            self.params['t2star'] = t2star
        if temp:
            self.params['temp'] = temp

    def get_factor(self):
        pi = tf.constant(np.pi, dtype=tf.float64)
        phi = tf.cast(self.params['phi'].get_value(), tf.float64)
        phi_0 = tf.cast(self.params['phi_0'].get_value(), tf.float64)
        if 'd' in params:
            d = tf.cast(self.params['d'].get_value(), tf.float64)
        elif 'gamma' in params:
            gamma = tf.cast(self.params['gamma'].get_value(), tf.complex128)
            d = (gamma - 1) / (gamma + 1)
        else:
            d = 0
        factor = tf.sqrt(tf.sqrt(
            tf.cos(pi * phi / phi_0)**2 + d**2 * tf.sin(pi * phi / phi_0)**2
        ))
        factor = tf.cast(factor, tf.complex128)
        return factor

    def get_anhar(self):
        anhar = tf.cast(self.params['anhar'].get_value(), tf.complex128)
        return anhar

    def get_freq(self):
        freq = tf.cast(self.params['freq'].get_value(), tf.complex128)
        anhar = tf.cast(self.params['anhar'].get_value(), tf.complex128)
        biased_freq = (freq - anhar) * self.get_factor() + anhar
        return biased_freq

    def init_Hs(self, ann_oper):
        resonator = hamiltonians["resonator"]
        self.Hs["freq"] = tf.Variable(resonator(ann_oper), dtype=tf.complex128)
        if self.hilbert_dim > 2:
            duffing = hamiltonians["duffing"]
            self.Hs["anhar"] = tf.Variable(duffing(ann_oper), dtype=tf.complex128)

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

    def get_Hamiltonian(self):
        h = self.get_freq() * self.Hs['freq']
        if self.hilbert_dim > 2:
            h += self.get_anhar() * self.Hs['anhar']
        return h

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
                if self.params['temp'].get_value().numpy():
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
        if Ls == []:
            raise Exception("No T1 or T2 provided")
        return tf.cast(sum(Ls), tf.complex128)


@dev_reg_deco
class SNAIL(PhysicalComponent):
    """
    Represents the element in a chip functioning as a three wave mixing element also knwon as a SNAIL.
    Reference: https://arxiv.org/pdf/1702.00869.pdf
    Parameters
    ----------
    freq: np.float64
        frequency of the qubit
    anhar: np.float64
        anharmonicity of the qubit. defined as w01 - w12
    beta: np.float64
        third order non_linearity of the qubit. 
    t1: np.float64
        t1, the time decay of the qubit due to dissipation
    t2star: np.float64
        t2star, the time decay of the qubit due to pure dephasing
    temp: np.float64
        temperature of the qubit, used to determine the Boltzmann distribution
        of energy level populations
    Class is mostly an exact copy of the Qubit class. The only difference is the added third order non linearity with a prefactor beta.
    The only modification is the get hamiltonian and init hamiltonian definition. Also imported the necessary third order non linearity
    from the hamiltonian library. 
    """
    def __init__(
        self,
        name: str,
        desc: str = " ",
        comment: str = " ",
        hilbert_dim: int = 4,
        freq: np.float64 = 0.0,
        anhar: np.float64 = 0.0,
        beta: np.float64 = 0.0,
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
        self.params['beta'] = beta
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
        Initialize the SNAIL Hamiltonians.
        Parameters
        ----------
        ann_oper : np.array
            Annihilation operator in the full Hilbert space
        """
        resonator = hamiltonians["resonator"]
        self.Hs["freq"] = tf.Variable(resonator(ann_oper), dtype=tf.complex128)
        if self.hilbert_dim > 2:
            duffing = hamiltonians["duffing"]
            self.Hs["anhar"] = tf.Variable(duffing(ann_oper), dtype=tf.complex128)
        third = hamiltonians["third_order"]
        self.Hs['beta'] = tf.Variable(third(ann_oper), dtype=tf.complex128)


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
        h += tf.cast(
                self.params['beta'].get_value(),
                tf.complex128
        ) * self.Hs['beta']
        if self.hilbert_dim > 2:
            h += tf.cast(
                self.params['anhar'].get_value(),
                tf.complex128
            ) * self.Hs['anhar']

        return h
    
    init_Ls = Qubit.__dict__['init_Ls']
    get_Lindbladian = Qubit.__dict__['get_Lindbladian'] 
    
    
@dev_reg_deco
class LineComponent(C3obj):
    """
    Represents the components connecting chip elements and drives.

    Parameters
    ----------
    connected: list
        specifies the component that are connected with this line

    """

    def __init__(self, **props):
        h_func = props.pop("hamiltonian_func")
        self.connected = props.pop("connected")
        if callable(h_func):
            self.hamiltonian_func = h_func
        else:
            self.hamiltonian_func = hamiltonians[h_func]
        super().__init__(**props)
        self.Hs = {}

    def asdict(self) -> dict:
        params = {}
        for key, item in self.params.items():
            params[key] = item.asdict()
        return {
            "c3type": self.__class__.__name__,
            "params": params,
            "hamiltonian_func": self.hamiltonian_func.__name__,
            "connected": self.connected,
        }


@dev_reg_deco
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
        name,
        desc=None,
        comment=None,
        strength=None,
        connected=None,
        params=None,
        hamiltonian_func=None,
    ):
        super().__init__(
            name=name,
            desc=desc,
            comment=comment,
            params=params,
            connected=connected,
            hamiltonian_func=hamiltonian_func,
        )
        if strength:
            self.params["strength"] = strength

    def init_Hs(self, opers_list):
        self.Hs["strength"] = tf.Variable(
            self.hamiltonian_func(opers_list), dtype=tf.complex128
        )

    def get_Hamiltonian(self):
        strength = tf.cast(self.params["strength"].get_value(), tf.complex128)
        return strength * self.Hs["strength"]


@dev_reg_deco
class Drive(LineComponent):
    """
    Represents a drive line.

    Parameters
    ----------
    connected: list
        all physical components receiving driving signals via this line

    """

    def init_Hs(self, ann_opers: list):
        hs = []
        for a in ann_opers:
            hs.append(tf.Variable(self.hamiltonian_func(a), dtype=tf.complex128))
        self.h = sum(hs)

    def get_Hamiltonian(self):
        return self.h
