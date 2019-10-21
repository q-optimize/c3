"""Component class and subclasses."""

import types
import numpy as np


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


class ControlComponent(C3obj):
    """
    Represents the components making up a pulse.

    Parameters
    ----------
    parameters: dict
        dictionary of the parameters needed for the shape-function to
        create the desired pulse
    bounds: dict
        boundaries of the parameters, i.e. technical limits of experimental
        setup or physical boundaries. needed for optimizer

    """

    def __init__(
            self,
            name: str,
            desc: str = " ",
            comment: str = " ",
            params: dict = {},
            bounds: dict = {},
            ):
        super().__init__(
            name=name,
            desc=desc,
            comment=comment
            )
        self.params = params
        self.bounds = bounds
        # check that the parameters and bounds have the same key
        if params.keys() != bounds.keys():
            raise ValueError('params and bounds must have same keys')


class Envelope(ControlComponent):
    """
    Represents the envelopes shaping a pulse.

    Parameters
    ----------
    shape: function
        function evaluating the shape in time

    """

    def __init__(
            self,
            name: str,
            desc: str = " ",
            comment: str = " ",
            params: dict = {},
            bounds: dict = {},
            shape: types.FunctionType = None,
            ):
        super().__init__(
            name=name,
            desc=desc,
            comment=comment,
            params=params,
            bounds=bounds,
            )
        self.shape = shape

    def get_shape_values(self, ts):
        """Return the value of the shape function at the specified times."""
        return self.shape(ts, self.params)


class Carrier(ControlComponent):
    """Represents the carrier of a pulse."""

    def __init__(
            self,
            name: str,
            desc: str = " ",
            comment: str = " ",
            params: dict = {},
            bounds: dict = {},
            ):
        super().__init__(
            name=name,
            desc=desc,
            comment=comment,
            params=params,
            bounds=bounds,
            )


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
            strength: np.float64 = 0.0
            ):
        super().__init__(
            name=name,
            desc=desc,
            comment=comment,
            connected=connected
            )
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
