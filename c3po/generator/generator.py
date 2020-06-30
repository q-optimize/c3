"""Singal generation stack."""

import copy
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from c3po.signal.gates import Instruction


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
            if "fluxbias" in self.devices:
                fluxbias = self.devices["fluxbias"]
            t_start = instr.t_start
            t_end = instr.t_end
            for chan in instr.comps:
                gen_signal[chan] = {}
                components = instr.comps[chan]
                lo_signal, omega_lo = lo.create_signal(components, t_start, t_end)
                awg_signal = awg.create_IQ(chan, components, t_start, t_end)
                flat_signal = dig_to_an.resample(awg_signal, t_start, t_end)
                if "resp" in self.devices:
                    conv_signal = resp.process(flat_signal)
                else:
                    conv_signal = flat_signal
                signal = mixer.combine(lo_signal, conv_signal)
                if "fluxbias" in self.devices and chan == "TC":
                    signal = fluxbias.frequency(signal)
                else:
                    signal = v_to_hz.transform(signal, omega_lo)
                gen_signal[chan]["values"] = signal
                gen_signal[chan]["ts"] = lo_signal['ts']
                # plt.figure()
                # plt.plot(awg.ts/1e-9, awg_signal['inphase']/1e-3, 'x', label='AWG', color="tab:red")
                # # plt.plot(awg.ts, awg_signal['quadrature'], 'xr')
                # plt.plot(lo_signal['ts']/1e-9, flat_signal['inphase']/1e-3, '-', label='interp', color="tab:blue")
                # plt.xlabel("Time[ns]")
                # plt.ylabel("Pulse amplitude[mV]")
                # plt.grid()
                # plt.savefig("/home/users/niwitt/awg.png", dpi=300)
                # plt.figure()
                # # plt.plot(lo_signal['ts'], flat_signal['quadrature'], 'r-')
                # plt.plot(lo_signal['ts']/1e-9, conv_signal['inphase']/1e-3, 'g-', label='convolved')
                # plt.xlabel("Time[ns]")
                # plt.ylabel("Pulse amplitude[mV]")
                # plt.grid()
                # plt.savefig("/home/users/niwitt/awg_smooth.png", dpi=300)
                # plt.figure()
                # plt.plot(awg.ts/1e-9, awg_signal['inphase']/1e-3, 'x', label='AWG', color="tab:red")
                # # plt.plot(awg.ts, awg_signal['quadrature'], 'xr')
                # plt.plot(lo_signal['ts']/1e-9, flat_signal['inphase']/1e-3, '-', label='interp', color="tab:blue")
                # # plt.plot(lo_signal['ts'], flat_signal['quadrature'], 'r-')
                # plt.plot(lo_signal['ts']/1e-9, conv_signal['inphase']/1e-3, 'g-', label='convolved')
                # plt.xlabel("Time[ns]")
                # plt.ylabel("Pulse amplitude[mV]")
                # plt.grid()
                # plt.legend(
                #     ["AWG samples", "upsampled", "convolution"]
                # )
                # plt.savefig("/home/users/niwitt/awg_combined.png", dpi=300)
                # plt.figure()
                # # plt.plot(lo_signal['ts'], conv_signal['quadrature'], 'y*:')
                # plt.plot(lo_signal['ts']/1e-9, signal/1e6, '-')
                # plt.title("Mixed with local oscillator")
                # plt.xlabel("Time[ns]")
                # plt.ylabel("Pulse amplitude[MHz]")
                # plt.grid()
                # plt.savefig("/home/users/niwitt/iq_mixer.png", dpi=300)
                # plt.show()
        self.signal = gen_signal
        # TODO clean up output here: ts is redundant
        return gen_signal, lo_signal['ts']

    def readout_signal(self, phase):
        return self.devices["readout"].readout(phase)
