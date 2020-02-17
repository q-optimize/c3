"""Singal generation stack."""

import copy
import numpy as np
import tensorflow as tf
from c3po.control import Instruction

class Generator:
    """Generator, creates signal from digital to what arrives to the chip."""

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
            t_start = instr.t_start
            t_end = instr.t_end
            for chan in instr.comps:
                gen_signal[chan] = {}
                channel = instr.comps[chan]
                lo_signal, omega_lo = lo.create_signal(channel, t_start, t_end)
                awg_signal, freq_offset = awg.create_IQ(channel, t_start, t_end)
                flat_signal = dig_to_an.resample(awg_signal, t_start, t_end)
                if "resp" in self.devices:
                    conv_signal = resp.process(flat_signal)
                else:
                    conv_signal = flat_signal
                signal = mixer.combine(lo_signal, conv_signal)
                signal = v_to_hz.transform(signal, omega_lo+freq_offset)

                gen_signal[chan]["values"] = signal
                gen_signal[chan]["ts"] = lo_signal['ts']
              #  plt.figure()
              #  plt.plot(awg.ts, awg_signal['inphase'], 'xb', label='AWG')
              #  plt.plot(awg.ts, awg_signal['quadrature'], 'xr')
              #  plt.plot(lo_signal['ts'], flat_signal['inphase'], 'b-', label='interp')
              #  plt.plot(lo_signal['ts'], flat_signal['quadrature'], 'r-')
              #  plt.plot(lo_signal['ts'], conv_signal['inphase'], 'g*:', label='convolved')
              #  plt.plot(lo_signal['ts'], conv_signal['quadrature'], 'y*:')
              #  plt.show()
              #  plt.figure()
              #  plt.plot(lo_signal['ts'], signal, '-')
              #  plt.title("Multiplex")
              #  plt.show()
        self.signal = gen_signal
        # TODO clean up output here: ts is redundant
        return gen_signal, lo_signal['ts']

    def readout_signal(self, phase):
        return self.devices["readout"].readout(phase)
