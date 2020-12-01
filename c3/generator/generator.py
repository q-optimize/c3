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
import tensorflow as tf
import matplotlib.pyplot as plt
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
            devices: list,
            resolution: np.float64 = 0.0
    ):
        # TODO consider making the dict into a list of devices
        # TODO check that you get at least 1 set of LO, AWG and mixer.
        self.devices = {}
        for dev in devices:
            self.devices[dev.name] = dev

        self.resolution = resolution
        # TODO add line knowledge (mapping of which devices are connected)

    def write_config(self):
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
        Perform the signal chain for a specified instruction, including local oscillator, AWG generation and IQ mixing.

        Parameters
        ----------
        instr : Instruction
            Operation to be performed, e.g. logical gate.

        Returns
        -------
        dict
            Signal to be applied to the physical device.

        """
        # TODO deal with multiple instructions within GateSet
        with tf.name_scope('Signal_generation'):
            gen_signal = {}
            lo = self.devices["lo"]
            awg = self.devices["awg"]
            # TODO make mixer optional and have a signal chain (eg Flux tuning)
            mixer = self.devices["mixer"]
            v_to_hz = self.devices["v_to_hz"]
            dig_to_an = self.devices["dac"]
            if "resp" in self.devices:
                resp = self.devices["resp"]
            if "highpass" in self.devices:
                highpass = self.devices["highpass"]
            if "fluxbias" in self.devices:
                fluxbias = self.devices["fluxbias"]
            if "lo_noise" in self.devices:
                lo_noise = self.devices["lo_noise"]
            if "pink_noise" in self.devices:
                pink_noise = self.devices["pink_noise"]
            if "flux_noise" in self.devices:
                flux_noise = self.devices["flux_noise"]
            if "awg_pink_noise" in self.devices:
                awg_pink_noise = self.devices["awg_pink_noise"]
            if "awg_noise" in self.devices:
                awg_noise = self.devices["awg_noise"]
            if "signal_noise" in self.devices:
                signal_noise = self.devices["signal_noise"]
            if "dc_noise" in self.devices:
                dc_noise = self.devices["dc_noise"]
            t_start = instr.t_start
            t_end = instr.t_end
            for chan in instr.comps:
                gen_signal[chan] = {}
                components = instr.comps[chan]
                lo_signal, omega_lo = lo.create_signal(components, t_start, t_end)
                awg_signal = awg.create_IQ(chan, components, t_start, t_end)
                if "awg_noise" in self.devices:
                    awg_signal = awg_noise.distort(awg_signal)
                flat_signal = dig_to_an.resample(awg_signal, t_start, t_end)
                if "resp" in self.devices:
                    conv_signal = resp.process(flat_signal)
                else:
                    conv_signal = flat_signal
                if "awg_pink_noise" in self.devices:
                    conv_signal = awg_pink_noise.distort(conv_signal)
                if "highpass" in self.devices:
                    conv_signal = highpass.process(conv_signal)
                if "lo_noise" in self.devices:
                    lo_signal = lo_noise.distort(lo_signal)
                signal = mixer.combine(lo_signal, conv_signal)
                if "signal_noise" in self.devices:
                    signal = signal_noise.distort(signal)
                if "fluxbias" in self.devices and (chan == "TC" or chan == "FluxDrive"):
                    if "dc_noise" in self.devices:
                        signal = dc_noise.distort(signal)
                    if "flux_noise" in self.devices:
                        signal = flux_noise.distort(signal)
                    signal = fluxbias.frequency(signal)
                else:
                    signal = v_to_hz.transform(signal, omega_lo)
                gen_signal[chan]["values"] = signal
                gen_signal[chan]["ts"] = lo_signal['ts']
        self.signal = gen_signal
        # TODO clean up output here: ts is redundant
        return gen_signal, lo_signal['ts']
