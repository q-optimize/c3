"""Singal generation stack."""

import types
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from c3po.component import C3obj
from c3po.control import Instruction, GateSet, Envelope, Carrier


class Generator:
    """Generator, creates signal from digital to what arrives to the chip."""

    def __init__(
            self,
            devices: dict,
            resolution: np.float64 = 0.0
    ):
        # TODO consider making the dict into a list of devices
        # TODO check that you get at least 1 set of LO, AWG and mixer.
        self.devices = devices
        self.resolution = resolution
        # TODO add line knowledge (mapping of which devices are connected)

    def generate_signals(self, instr: Instruction):
        # TODO deal with multiple instructions within GateSet
        with tf.name_scope('Signal_generation'):
            gen_signal = {}
            lo = self.devices["lo"]
            awg = self.devices["awg"]
            # TODO make mixer optional and have a signal chain (eg Flux tuning)
            mixer = self.devices["mixer"]
            v_to_hz = self.devices["v_to_hz"]
            t_start = instr.t_start
            t_end = instr.t_end
            for chan in instr.comps:
                gen_signal[chan] = {}
                channel = instr.comps[chan]
                lo_signal = lo.create_signal(channel, t_start, t_end)
                awg_signal = awg.create_IQ(channel, t_start, t_end)
                signal = mixer.combine(lo_signal, awg_signal)
                signal = v_to_hz.transform(signal)
                gen_signal[chan]["values"] = signal
                gen_signal[chan]["ts"] = lo_signal['ts']

        self.signal = gen_signal
        return gen_signal


class Device(C3obj):
    """Device that is part of the stack generating the instruction signals."""

    def __init__(
            self,
            name: str = " ",
            desc: str = " ",
            comment: str = " ",
            resolution: np.float64 = 0.0
    ):
        super().__init__(
            name=name,
            desc=desc,
            comment=comment
        )
        self.resolution = resolution
        self.params = {}

    def prepare_plot(self):
        plt.rcParams['figure.dpi'] = 100
        fig, axs = plt.subplots(1, 1)
        self.fig = fig
        self.axs = axs
        # return self.fig, self.axs

    def calc_slice_num(
            self,
            t_start: np.float64,
            t_end: np.float64
            ):
        res = self.resolution
        self.slice_num = int(np.abs(t_start - t_end) * res)
        # return self.slice_num

    def create_ts(
            self,
            t_start: np.float64,
            t_end: np.float64,
            centered: bool = True
            ):
        if not hasattr(self, 'slice_num'):
            self.calc_slice_num(t_start, t_end)
        dt = 1 / self.resolution
        if centered:
            offset = dt/2
            num = self.slice_num
        else:
            offset = 0
            num = self.slice_num + 1
        t_start = tf.constant(t_start + offset, dtype=tf.float64)
        t_end = tf.constant(t_end - offset, dtype=tf.float64)
        ts = tf.linspace(t_start, t_end, num)
        return ts

    def get_parameters(self):
        params = []
        for key in sorted(self.params.keys()):
            params.append(self.params[key])
        return params

    def set_parameters(self, values):
        idx = 0
        for key in sorted(self.params.keys()):
            self.params[key] = values[idx]
            idx += 1

    def list_parameters(self):
        par_list = []
        for par_key in sorted(self.params.keys()):
            par_id = (self.name, par_key)
            par_list.append(par_id)
        return par_list


class Volts_to_Hertz(Device):
    """Upsacle the voltage signal to an amplitude to plug in the model."""

    def __init__(
            self,
            name: str = " ",
            desc: str = " ",
            comment: str = " ",
            resolution: np.float64 = 0.0,
            V_to_Hz: np.float64 = 1.0
    ):
        super().__init__(
            name=name,
            desc=desc,
            comment=comment,
        )
        self.signal = None
        self.params['V_to_Hz'] = V_to_Hz

    def transform(self, mixed_signal):
        """Transform signal from value of V to Hz."""
        self.signal = mixed_signal * self.params['V_to_Hz']
        return self.signal


class Filter(Device):
    """Apply a filter function to the signal."""

    def __init__(
            self,
            name: str = " ",
            desc: str = " ",
            comment: str = " ",
            resolution: np.float64 = 0.0,
            filter_fuction: types.FunctionType = None,
    ):
        super().__init__(
            name=name,
            desc=desc,
            comment=comment,
        )
        self.filter_fuction = filter_fuction
        self.signal = None

    def filter(self, Hz_signal):
        """Apply a filter function to the signal."""
        self.signal = self.filter_fuction(Hz_signal)
        return self.signal


class Transfer(Device):
    """Apply a transfer function to the signal."""

    def __init__(
            self,
            name: str = " ",
            desc: str = " ",
            comment: str = " ",
            resolution: np.float64 = 0.0,
            transfer_fuction: types.FunctionType = None,
    ):
        super().__init__(
            name=name,
            desc=desc,
            comment=comment,
        )
        self.transfer_fuction = transfer_fuction
        self.signal = None

    def transfer(self, filter_signal):
        """Apply a transfer function to the signal."""
        self.signal = self.transfer_fuction(filter_signal)
        return self.signal


class Mixer(Device):
    """Mixer device, combines inputs from the local oscillator and the AWG."""

    def __init__(
            self,
            name: str = " ",
            desc: str = " ",
            comment: str = " "
    ):
        super().__init__(
            name=name,
            desc=desc,
            comment=comment,
        )

        self.signal = None

    def combine(self, lo_signal, awg_signal):
        """Combine signal from AWG and LO."""
        cos, sin = lo_signal["values"]
        ts = lo_signal['ts']
        old_dim = awg_signal["inphase"].shape[0]
        new_dim = ts.shape[0]
        inphase = tf.reshape(
                    tf.image.resize(
                        tf.reshape(
                            awg_signal["inphase"],
                            shape=[1, old_dim, 1]),
                        size=[1, new_dim],
                        method='nearest'),
                    shape=[new_dim])
        self.inphase = inphase
        quadrature = tf.reshape(
                        tf.image.resize(
                            tf.reshape(
                                awg_signal["quadrature"],
                                shape=[1, old_dim, 1]),
                            size=[1, new_dim],
                            method='nearest'),
                        shape=[new_dim])
        self.signal = (inphase * cos + quadrature * sin)
        return self.signal


class LO(Device):
    """Local oscillator device, generates a constant oscillating signal."""

    def __init__(
        self,
        name: str = " ",
        desc: str = " ",
        comment: str = " ",
        resolution: np.float64 = 0.0
    ):
        super().__init__(
            name=name,
            desc=desc,
            comment=comment,
            resolution=resolution
        )

        self.signal = {}

    def create_signal(self, channel: dict, t_start: float, t_end: float):
        # TODO check somewhere that there is only 1 carrier per instruction
        ts = self.create_ts(t_start, t_end, centered=True)
        for c in channel:
            comp = channel[c]
            if isinstance(comp, Carrier):
                omega_lo = comp.params['freq']
                self.signal["values"] = (
                    tf.cos(omega_lo * ts), tf.sin(omega_lo * ts)
                )
                self.signal["ts"] = ts
        return self.signal


class AWG(Device):
    """AWG device, transforms digital input to analog signal."""

    def __init__(
            self,
            name: str = " ",
            desc: str = " ",
            comment: str = " ",
            resolution: np.float64 = 0.0,
            ):
        super().__init__(
            name=name,
            desc=desc,
            comment=comment,
            resolution=resolution
            )

        self.options = ""
        # TODO move the options pwc & drag to the instruction object
        self.signal = {}
        self.amp_tot_sq = None

# TODO create DC function

    def create_IQ(self, channel: dict, t_start: float, t_end: float):
        """
        Construct the in-phase (I) and quadrature (Q) components of the signal.

        These are universal to either experiment or simulation.
        In the experiment these will be routed to AWG and mixer
        electronics, while in the simulation they provide the shapes of the
        instruction fields to be added to the Hamiltonian.

        """
        with tf.name_scope("I_Q_generation"):
            ts = self.create_ts(t_start, t_end, centered=True)
            self.ts = ts
            dt = ts[1] - ts[0]
            amp_tot_sq = 0.0
            inphase_comps = []
            quadrature_comps = []

            if (self.options == 'pwc'):
                for key in channel:
                    comp = channel[key]
                    if isinstance(comp, Envelope):
                        norm = 1
                        inphase = comp.params['inphase']
                        quadrature = comp.params['quadrature']

            elif (self.options == 'drag'):
                for key in channel:
                    comp = channel[key]
                    if isinstance(comp, Envelope):

                        amp = comp.params['amp']
                        amp_tot_sq += amp**2

                        xy_angle = comp.params['xy_angle']
                        freq_offset = comp.params['freq_offset']
                        delta = comp.params['delta'] * dt
                        # TODO Deal with the scale of delta

                        with tf.GradientTape() as t:
                            t.watch(ts)
                            env = comp.get_shape_values(ts)

                        denv = t.gradient(env, ts)
                        if denv is None:
                            denv = tf.zeros_like(ts, dtype=tf.float64)
                        phase = xy_angle - freq_offset * ts
                        inphase_comps.append(
                            amp * (
                                env * tf.cos(phase)
                                + denv * delta * tf.sin(phase)
                            )
                        )
                        quadrature_comps.append(
                            amp * (
                                env * tf.sin(phase)
                                - denv * delta * tf.cos(phase)
                            )
                        )
                norm = tf.sqrt(tf.cast(amp_tot_sq, tf.float64))
                inphase = tf.add_n(inphase_comps, name="inphase")
                quadrature = tf.add_n(quadrature_comps, name="quadrature")

            else:
                for key in channel:
                    comp = channel[key]
                    if isinstance(comp, Envelope):

                        amp = comp.params['amp']

                        amp_tot_sq += amp**2

                        xy_angle = comp.params['xy_angle']
                        freq_offset = comp.params['freq_offset']
                        inphase_comps.append(
                            amp * comp.get_shape_values(ts)
                            * tf.cos(xy_angle - freq_offset * ts)
                            )
                        quadrature_comps.append(
                            amp * comp.get_shape_values(ts)
                            * tf.sin(xy_angle - freq_offset * ts)
                            )

                norm = tf.sqrt(tf.cast(amp_tot_sq, tf.float64))
                inphase = tf.add_n(inphase_comps, name="inphase")
                quadrature = tf.add_n(quadrature_comps, name="quadrature")

        self.amp_tot = norm
        self.signal['inphase'] = inphase / norm
        self.signal['quadrature'] = quadrature / norm
        return {"inphase": inphase, "quadrature": quadrature}
        # TODO decide when and where to return/sotre params scaled or not

    def get_I(self):
        return self.amp_tot * self.signal['inphase']

    def get_Q(self):
        return self.amp_tot * self.signal['quadrature']

    def plot_IQ_components(self, instruction: Instruction):
        """Plot instruction functions."""
        ts = self.create_ts(instruction.t_start, instruction.t_end)
        inphase = self.get_I()
        quadrature = self.get_Q()

        if not hasattr(self, 'fig') or not hasattr(self, 'axs'):
            self.prepare_plot()
        fig = self.fig
        ax = self.axs

        ax.clear()
        ax.plot(ts/1e-9, inphase/1e-3)
        ax.plot(ts/1e-9, quadrature/1e-3)
        ax.grid()
        ax.legend(['I', 'Q'])
        ax.set_xlabel('Time [ns]')
        ax.set_ylabel('Amplitude [mV]')
        plt.show()
        fig.canvas.draw()
        fig.canvas.flush_events()
