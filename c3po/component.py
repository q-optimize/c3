"""Component class and subclasses."""

import types
import numpy as np
import tensorflow as tf
from c3po.utils import num3str


class Quantity:
    """
    Represents any parameter used in the model or the pulse speficiation. For
    arithmetic operations just the numeric value is used.

    Parameters
    ----------
    value: np.array(np.float64) or np.float64
        value of the quantity
    min: np.array(np.float64) or np.float64
        minimun values this quantity is allowed to take
    max: np.array(np.float64) or np.float64
        maximum values this quantity is allowed to take
    symbol: str
        latex representation
    unit: str
        physical unit

    """

    def __init__(
        self,
        # TODO how to specify two options for type
        value,
        min,
        max,
        symbol: str = '\\alpha',
        unit: str = 'a.u.'
    ):
        self.offset = min
        self.scale = np.array(np.abs(max - min))
        self.set_value(value)
        self.symbol = symbol
        self.unit = unit

    def __add__(self, other):
        return self.get_value() + other

    def __radd__(self, other):
        return self.get_value() + other

    def __sub__(self, other):
        return self.get_value() - other

    def __rsub__(self, other):
        return other - self.get_value()

    def __mul__(self, other):
        return self.get_value() * other

    def __rmul__(self, other):
        return self.get_value() * other

    def __pow__(self, other):
        return self.get_value() ** other

    def __rpow__(self, other):
        return other ** self.get_value()

    def __truediv__(self, other):
        return self.get_value() / other

    def __rtruediv__(self, other):
        return other / self.get_value()

    def __str__(self):
        val = self.get_value()
        if self.unit == "Hz 2pi":
            val = val / 2 / np.pi
        return num3str(val) + self.unit

    def __float__(self):
        return self.get_value()

    def get_value(self, val=None):
        if val is None:
            val = self.value.numpy()
        return self.scale * (val + 1) / 2 + self.offset

    def set_value(self, val):
        # setting can be numpyish
        tmp = 2 * (np.array(val) - self.offset) / self.scale - 1
        if np.any(tmp < -1) or np.any(tmp > 1):
            raise Exception(f"Value {val} out of bounds for quantity {self}.")
            # TODO if we want we can extend bounds when force flag is given
        else:
            self.value = tf.constant(tmp, dtype=tf.float64)

    def tf_get_value(self):
        # getting needs to be tensorflowy
        return tf.cast(
            self.scale * (self.value + 1) / 2 + self.offset,
            tf.float64
        )

    def tf_set_value(self, val):
        self.value = tf.acos(tf.cos((val + 1) * np.pi / 2)) / np.pi * 2 - 1

    # TODO find good name for physical_bounded vs order1_unbounded
    # TODO At some point make self.value private


class C3obj:
    """
    Represents an abstract object.

    Parameters
    ----------
    name: str
        short name that will be used as identifier
    desc: str
        longer description of the component
    comment: str
        additional information about the component

    """

    def __init__(
            self,
            name: str,
            desc: str = " ",
            comment: str = " "
            ):
        self.name = name
        self.desc = desc
        self.comment = comment


class PhysicalComponent(C3obj):
    """
    Represents the components making up a chip.

    Parameters
    ----------
    hilber_dim: int
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
        self.values = {}

    def get_values(self):
        """Return values of the physical component."""
        return self.values


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
        self.values['freq'] = freq
        if hilbert_dim > 2:
            self.values['anhar'] = anhar
        if t1:
            self.values['t1'] = t1
        if t2star:
            self.values['t2star'] = t2star
        if temp:
            self.values['temp'] = temp


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
        self.values['freq'] = freq


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
            ):
        super().__init__(
            name=name,
            desc=desc,
            comment=comment
            )
        self.connected = connected
        self.values = {}


class Coupling(LineComponent):
    """
    Represents a coupling behaviour between elements.

    Parameters
    ----------
    strength: np.float64
        coupling strenght
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
            hamiltonian: types.FunctionType = None,
            ):
        super().__init__(
            name=name,
            desc=desc,
            comment=comment,
            connected=connected
            )
        self.hamiltonian = hamiltonian
        self.values['strength'] = strength


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
            hamiltonian: types.FunctionType = None,
            ):
        super().__init__(
            name=name,
            desc=desc,
            comment=comment,
            connected=connected
            )
        self.hamiltonian = hamiltonian
