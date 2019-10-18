import uuid

# this is originally the class "Pulse"
class Component:

    def __init__(
            self,
            name : string = " ",
            desc : string = " ",
            comment : string = " ",
            ):

        self.name = name
        self.desc = desc
        self.comment = comment


class ControlComponent(Component):
    """Represents a pulse.

    Parameters
    ----------
    shape : func
        Function handle to function specifying the exact shape of the pulse
    parameters : dict
        dictionary of the parameters needed for the shape-function to
        create the desired pulse
    bounds : dict
        boundaries of the parameters, i.e. technical limits of experimental
        setup or physical boundaries. needed for optimizer
    """

    def __init__(
            self,
            name : string = " ",
            desc : string = " ",
            comment : string = " ",
            params : dict = {},
            bounds : dict = {},
            groups : list = [],
            shape = None
            ):
        super().__init__(
            name = name,
            desc = desc,
            comment = comment
            )
        self.params = params
        self.bounds = bounds
        self.groups = groups
        self.shape = shape


    def get_shape_values(self, ts):
        return self.shape(ts, self.params)


class Envelope(ControlComponent):
    pass


class Carrier(ControlComponent):
    pass


class PhysicalComponent(Component):
    def __init__(
            self,
            name : string = " ",
            desc : string = " ",
            comment : string = " ",
            hilbert_dim : int = 0,
            ):
        super().__init__(
            name = name,
            desc = desc,
            comment = comment
            )
        self.hilbert_dim = hilbert_dim
        self.values = {}

    def get_values(self):
        return self.values


class Qubit(PhysicalComponent):
    def __init__(
            self,
            name : string = " ",
            desc : string = " ",
            comment : string = " ",
            hilbert_dim = None,
            freq = None,
            delta = None,
            T1 = None,
            T2star = None,
            temp = None
            ):
        super().__init__(
            name = name,
            desc = desc,
            comment = comment,
            hilbert_dim = hilbert_dim
            )
        self.values['freq'] = freq
        if hilbert_dim > 2:
            self.values['delta'] = delta
        if T1: self.values['T1'] = T1
        if T2star: self.values['T2star'] = T2star
        if temp: self.values['temp'] = temp

class Resonator(PhysicalComponent):
    def __init__(
            self,
            name : string = " ",
            desc : string = " ",
            comment : string = " ",
            hilbert_dim = None,
            freq = None
            ):
        super().__init__(
            name = name,
            desc = desc,
            comment = comment,
            hilbert_dim = hilbert_dim
            )
        self.values['freq'] = freq

class Coupling(PhysicalComponent):
    def __init__(
            self,
            name : string = " ",
            desc : string = " ",
            comment : string = " ",
            hilbert_dim = None,
            connected = None,
            strength = None
            ):
        super().__init__(
            name = name,
            desc = desc,
            comment = comment,
            hilbert_dim = hilbert_dim
            )
        self.values['strength'] = strength
        self.connected = connected

class Drive(PhysicalComponent):
    def __init__(
            self,
            name : string = " ",
            desc : string = " ",
            comment : string = " ",
            Hamiltonian = None,
            hilbert_dim = None,
            connected = None,
            strength = None
            ):
        super().__init__(
            name = name,
            desc = desc,
            comment = comment,
            hilbert_dim = hilbert_dim
            )
        self.connected = connected
        self.Hamiltonian = Hamiltonian
