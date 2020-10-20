import copy
import numpy as np
from c3.signal.pulse import InstructionComponent


class Instruction():
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
            channels: list = [],
            t_start: np.float64 = 0.0,
            t_end: np.float64 = 0.0,
            ):
        self.name = name
        self.t_start = t_start
        self.t_end = t_end
        self.comps = {}
        for chan in channels:
            self.comps[chan] = {}
        # TODO remove redundancy of channels in instruction

    def write_config(self):
        cfg = copy.deepcopy(self.__dict__)
        for chan in self.comps:
            for comp in self.comps[chan]:
                cfg['comps'][chan][comp] = 0
        return cfg

    def add_component(self, comp: InstructionComponent, chan: str):
        """
        Add one component, e.g. an envelope, local oscillator, to a channel.

        Parameters
        ----------
        comp : InstructionComponent
            Component to be added.
        chan : str
            Identifier for the target channel

        """
        self.comps[chan][comp.name] = comp
