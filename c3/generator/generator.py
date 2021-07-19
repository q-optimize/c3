"""
Signal generation stack.

Contrary to most quanutm simulators, C^3 includes a detailed simulation of the control
stack. Each component in the stack and its functions are simulated individually and
combined here.

Example: A local oscillator and arbitrary waveform generator signal
are put through via a mixer device to produce an effective modulated signal.
"""

import copy
from typing import List, Callable
import hjson
import numpy as np
import tensorflow as tf
from c3.c3objs import hjson_decode, hjson_encode
from c3.signal.gates import Instruction
from c3.generator.devices import devices as dev_lib
from graphlib import TopologicalSorter


class Generator:
    """
    Generator, creates signal from digital to what arrives to the chip.

    Parameters
    ----------
    devices : list
        Physical or abstract devices in the signal processing chain.
    resolution : np.float64
        Resolution at which continuous functions are sampled.
    callback : Callable
        Function that is called after each device in the signal line.

    """

    def __init__(
        self,
        devices: dict = None,
        chains: dict = None,
        resolution: np.float64 = 0.0,
        callback: Callable = None,
    ):
        self.devices = {}
        if devices:
            self.devices = devices
        self.chains = {}
        self.sorted_chains: dict[str, List[str]] = {}
        if chains:
            self.chains = chains
            self.__check_signal_chains()
        self.resolution = resolution
        self.callback = callback

    def __check_signal_chains(self) -> None:
        for channel, chain in self.chains.items():
            signals = 0
            for device_id, sources in chain.items():
                # all source devices need to exist
                for dev in sources:
                    if dev not in self.devices:
                        raise Exception(f"C3:Error: device {dev} not found.")

                # the expected number of inputs must match the connected devices
                if self.devices[device_id].inputs != len(sources):
                    raise Exception(
                        f"C3:Error: device {device_id} expects {self.devices[device_id].inputs} inputs, but {len(sources)} found."
                    )

                # overall the chain should have exactly 1 output signal
                signals -= self.devices[device_id].inputs
                signals += self.devices[device_id].outputs
            if signals != 1:
                raise Exception(
                    "C3:ERROR: Signal chain for channel '"
                    + channel
                    + "' contains unmatched number of inputs and outputs."
                )

            # bring chain in topological order
            sorter = TopologicalSorter(chain)
            self.sorted_chains[channel] = list(sorter.static_order())

    def read_config(self, filepath: str) -> None:
        """
        Load a file and parse it to create a Generator object.

        Parameters
        ----------
        filepath : str
            Location of the configuration file

        """
        with open(filepath, "r") as cfg_file:
            cfg = hjson.loads(cfg_file.read(), object_pairs_hook=hjson_decode)
        self.fromdict(cfg)

    def fromdict(self, cfg: dict) -> None:
        for name, props in cfg["Devices"].items():
            props["name"] = name
            dev_type = props.pop("c3type")
            self.devices[name] = dev_lib[dev_type](**props)
        self.chains = cfg["Chains"]
        self.__check_signal_chains()

    def write_config(self, filepath: str) -> None:
        """
        Write dictionary to a HJSON file.
        """
        with open(filepath, "w") as cfg_file:
            hjson.dump(self.asdict(), cfg_file, default=hjson_encode)

    def asdict(self) -> dict:
        """
        Return a dictionary compatible with config files.
        """
        devices = {}
        for name, dev in self.devices.items():
            devices[name] = dev.asdict()
        return {"Devices": devices, "Chains": self.chains}

    def __str__(self) -> str:
        return hjson.dumps(self.asdict(), default=hjson_encode)

    def generate_signals(self, instr: Instruction) -> dict:
        """
        Perform the signal chain for a specified instruction, including local
        oscillator, AWG generation and IQ mixing.

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
            chain = self.chains[chan]

            # create list of succeeding devices
            successors = {}
            for dev_id in chain:
                successors[dev_id] = [x for x in chain if dev_id in chain[x]]

            signal_stack: dict[str, tf.constant] = {}
            for dev_id in self.sorted_chains[chan]:
                # collect inputs
                sources = self.chains[chan][dev_id]
                inputs = [signal_stack[x] for x in sources]

                # calculate the output and store it in the stack
                dev = self.devices[dev_id]
                output = dev.process(instr, chan, *inputs)
                signal_stack[dev_id] = output

                # remove inputs if they are not needed anymore
                for source in sources:
                    successors[source].remove(dev_id)
                    if len(successors[source]) < 1:
                        del signal_stack[source]

                # call the callback with the current signal
                if self.callback:
                    self.callback(chan, dev_id, output)

            gen_signal[chan] = copy.deepcopy(signal_stack[dev_id])

        # Hack to use crosstalk. Will be generalized to a post-processing module.
        # TODO: Rework of the signal generation for larger chips, similar to qiskit
        if "crosstalk" in self.devices:
            gen_signal = self.devices["crosstalk"].process(signal=gen_signal)
        return gen_signal
