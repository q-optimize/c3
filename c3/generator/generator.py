"""
Signal generation stack.

Contrary to most quanutm simulators, C^3 includes a detailed simulation of the control
stack. Each component in the stack and its functions are simulated individually and
combined here.

Example: A local oscillator and arbitrary waveform generator signal
are put through via a mixer device to produce an effective modulated signal.
"""

import copy
import hjson
import numpy as np
from c3.signal.gates import Instruction
from c3.generator.devices import devices as dev_lib


class Generator:
    """
    Generator, creates signal from digital to what arrives to the chip.

    Parameters
    ----------
    devices : list
        Physical or abstract devices in the signal processing chain.
    resolution : np.float64
        Resolution at which continuous functions are sampled.

    """

    def __init__(
            self,
            devices: dict = None,
            chain: list = None,
            resolution: np.float64 = 0.0
    ):
        self.devices = {}
        if devices:
            self.devices = devices
        self.chain = []
        if chain:
            self.chain = chain
            self.__check_signal_chain()
        self.resolution = resolution

    def __check_signal_chain(self) -> None:
        signals = 0
        for device_id in self.chain:
            signals -= self.devices[device_id].inputs
            signals += self.devices[device_id].outputs
        if signals != 1:
            raise Exception(
                "C3:ERROR: Signal chain contains unmatched number"
                " of inputs and outputs."
            )

    def read_config(self, filepath: str) -> None:
        """
        Load a file and parse it to create a Generator object.

        Parameters
        ----------
        filepath : str
            Location of the configuration file

        """
        with open(filepath, "r") as cfg_file:
            cfg = hjson.loads(cfg_file.read())
        for name, props in cfg["Devices"].items():
            props["name"] = name
            self.devices[name] = dev_lib[name](**props)
        self.chain = cfg["Chain"]
        self.__check_signal_chain()

    def generate_signals(self, instr: Instruction):
        """
        Perform the signal chain for a specified instruction, including local oscillator, AWG
        generation and IQ mixing.

        Parameters
        ----------
        instr : Instruction
            Operation to be performed, e.g. logical gate.

        Returns
        -------
        dict
            Signal to be applied to the physical device.

        """
        gen_signal = {}
        for chan in instr.comps:
            signal_stack = []
            for dev_id in self.chain:
                dev = self.devices[dev_id]
                inputs = []
                for input_num in range(dev.inputs):
                    inputs.append(signal_stack.pop())
                outputs = dev.process(instr, chan, *inputs)
                signal_stack.append(outputs)
             # The stack is reused here, thus we need to deepcopy.
            gen_signal[chan] = copy.deepcopy(signal_stack.pop())
        return gen_signal
