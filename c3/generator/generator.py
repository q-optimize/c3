"""
Signal generation stack.

Contrary to most quanutm simulators, C^3 includes a detailed simulation of the control
stack. Each component in the stack and its functions are simulated individually and
combined here.

Example: A local oscillator and arbitrary waveform generator signal
are put through via a mixer device to produce an effective modulated signal.
"""

import copy
import numpy as np
from c3.signal.gates import Instruction


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
            devices: dict,
            chain: list,
            resolution: np.float64 = 0.0
    ):
        self.devices = devices
        signals = 0
        for device_id in chain:
            signals -= devices[device_id].inputs
            signals += devices[device_id].outputs
        if signals != 0:
            raise Exception(
                "C3:ERROR: Signal chain contains unmatched number"
                " of inputs and outputs."
            )
        self.chain = chain
        self.resolution = resolution

    def write_config(self):
        """
        WIP Write current status to file.
        """
        cfg = {}
        cfg = copy.deepcopy(self.__dict__)
        devcfg = {}
        for key in self.devices:
            dev = self.devices[key]
            devcfg[dev.name] = dev.write_config()
        cfg["devices"] = devcfg
        cfg.pop('signal', None)
        return cfg

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
                for input_num in dev.inputs:
                    inputs.append(signal_stack.pop())
                outputs = dev.process(instr, *inputs)
                signal_stack.extend(outputs)
                gen_signal[chan] = signal_stack.pop()
        return gen_signal
