"""
Signal generation stack.

Contrary to most quanutm simulators, C^3 includes a detailed simulation of the control
stack. Each component in the stack and its functions are simulated individually and
combined here.

Example: A local oscillator and arbitrary waveform generator signal
are put through via a mixer device to produce an effective modulated signal.
"""

from typing import List, Callable, Dict
import hjson
import numpy as np
import tensorflow as tf
from c3.c3objs import hjson_decode, hjson_encode
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
        self.sorted_chains: Dict[str, List[str]] = {}
        if chains:
            self.chains = chains
            self.__check_signal_chains()
        self.resolution = resolution
        self.callback = callback

    def __check_signal_chains(self) -> None:
        for channel, chain in self.chains.items():
            signals = 0
            for device_id, sources in chain.items():
                # all source devices need to exist and have the same resolution
                if sources:
                    res = self.devices[sources[0]].resolution
                for dev in sources:
                    if dev not in self.devices:
                        raise Exception(f"C3:Error: device {dev} not found.")
                    if res != self.devices[dev].resolution:
                        raise Exception(
                            f"C3:Error: Different resolution of inputs in {channel} {device_id}:{sources}."
                        )

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
            self.sorted_chains[channel] = self.__topological_ordering(
                self.chains[channel]
            )

    def __topological_ordering(self, predecessors: Dict[str, List[str]]) -> List[str]:
        """
        Computes the topological ordering of a directed acyclic graph.

        Parameters
        ----------
        predecessors : dict
            list of preceding nodes for each node

        Returns
        -------
            a list of all nodes in topological ordering

        Raises
        ------
        ValueError
            if the graph contains a cycle
        """
        stack = [x for x in predecessors if len(predecessors[x]) == 0]
        num_sources = {node: len(predecessors[node]) for node in predecessors}
        successors = {}
        for node in predecessors:
            successors[node] = [x for x in predecessors if node in predecessors[x]]
        ordered = []

        while stack:
            src = stack.pop()
            for node in successors[src]:
                num_sources[node] -= 1
                if num_sources[node] == 0:
                    stack.append(node)
            ordered.append(src)

        if len(ordered) != len(successors):
            raise Exception("C3:ERROR: Device chain contains a cycle")
        return ordered

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
        gen_signal: Dict[str, Dict[str, tf.constant]] = {}
        signal_stack: Dict[str, Dict[str, tf.constant]] = {}
        for chan in instr.comps:
            chain = self.chains[chan]
            signal_stack[chan] = {}

            # create list of succeeding devices
            successors = {}
            for dev_id in chain:
                successors[dev_id] = [x for x in chain if dev_id in chain[x]]

            for dev_id in self.sorted_chains[chan]:
                # collect inputs
                sources = self.chains[chan][dev_id]
                inputs = [signal_stack[chan][x] for x in sources]

                # calculate the output and store it in the stack
                dev = self.devices[dev_id]
                output = dev.process(instr, chan, *inputs)
                signal_stack[chan][dev_id] = output

                # remove inputs if they are not needed anymore
                # for source in sources:
                #     successors[source].remove(dev_id)
                #     if len(successors[source]) < 1:
                #         del signal_stack[chan][source]

                # call the callback with the current signal
                if self.callback:
                    self.callback(chan, dev_id, output)

            gen_signal[chan] = {}
            for key in output.keys():
                gen_signal[chan][key] = tf.identity(output[key])

        # Hack to use crosstalk. Will be generalized to a post-processing module.
        # TODO: Rework of the signal generation for larger chips, similar to qiskit
        if "crosstalk" in self.devices:
            gen_signal = self.devices["crosstalk"].process(signal=gen_signal)
        return gen_signal
