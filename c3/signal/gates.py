import hjson
import numpy as np
from c3.c3objs import C3obj, Quantity
from c3.signal.pulse import Envelope, Carrier
from c3.libraries.envelopes import gaussian_nonorm
from c3.libraries.constants import GATES
from c3.utils.qt_utils import kron_ids


class Instruction:
    """
    Collection of components making up the control signal for a line.

    Parameters
    ----------
    t_start : np.float64
        Start of the signal.
    t_end : np.float64
        End of the signal.
    channels : list
        List of channel names (strings)


    Attributes
    ----------
    comps : dict
        Nested dictionary with lines and components as keys

    Example:
    comps = {
             'channel_1' : {
                            'envelope1': envelope1,
                            'envelope2': envelope2,
                            'carrier': carrier
                            }
             }

    """

    def __init__(
        self,
        name: str = " ",
        targets: list = [0],
        params: list = None,
        ideal: np.array = None,
        channels: list = [],
        t_start: np.float64 = 0.0,
        t_end: np.float64 = 0.0,
    ):
        self.name = name
        self.targets = targets
        self.params = params
        self.t_start = t_start
        self.t_end = t_end
        self.comps = {}  # type: ignore
        if ideal:
            self.ideal = ideal
        elif name in GATES.keys():
            self.ideal = GATES[name]
        else:
            self.ideal = None
        for chan in channels:
            self.comps[chan] = {}

    def as_openqasm(self) -> dict:
        asdict = {"name": self.name, "qubits": self.targets, "params": self.params}
        if self.ideal:
            asdict["ideal"] = self.ideal
        return asdict

    def get_ideal_gate(self, dims):
        if self.ideal is None:
            raise Exception(
                "C3:ERROR: No ideal representation definded for gate"
                f" {self.name+str(self.targets)}"
            )
        return kron_ids(
            [2] * len(dims),  # we compare to the computational basis
            self.targets,
            [self.ideal],
        )

    def asdict(self) -> dict:
        components = {}  # type:ignore
        for chan, item in self.comps.items():
            components[chan] = {}
            for key, comp in item.items():
                components[chan][key] = comp.asdict()
        return {"gate_length": self.t_end - self.t_start, "drive_channels": components}

    def __str__(self) -> str:
        return hjson.dumps(self.asdict())

    def add_component(self, comp: C3obj, chan: str) -> None:
        """
        Add one component, e.g. an envelope, local oscillator, to a channel.

        Parameters
        ----------
        comp : C3obj
            Component to be added.
        chan : str
            Identifier for the target channel

        """
        self.comps[chan][comp.name] = comp

    def quick_setup(self, chan, qubit_freq, gate_time, v2hz=1, sideband=None) -> None:
        """
        Initialize this instruction with a default envelope and carrier.
        """
        pi_half_amp = np.pi / 2 / gate_time / v2hz * 2 * np.pi
        env_params = {
            "t_final": Quantity(value=gate_time, unit="s"),
            "amp": Quantity(
                value=pi_half_amp, min_val=0.0, max_val=3 * pi_half_amp, unit="V"
            ),
        }
        carrier_freq = qubit_freq
        if sideband:
            env_params["freq_offset"] = Quantity(value=sideband, unit="Hz 2pi")
            carrier_freq -= sideband
        self.comps[chan]["gaussian"] = Envelope(
            "gaussian", shape=gaussian_nonorm, params=env_params
        )
        self.comps[chan]["carrier"] = Carrier(
            "Carr_" + chan, params={"freq": Quantity(value=carrier_freq, unit="Hz 2pi")}
        )
