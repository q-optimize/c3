"""Singal generation stack."""

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from c3po.component import C3obj, Envelope, Carrier
from c3po.control import Control, ControlSet


class Device(C3obj):
    """Device that is part of the stack generating the control signals."""

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
        dt = 1/self.resolution
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


class Mixer(Device):
    """Mixer device, combines inputs from the local oscillator and the AWG."""

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
        self.mixed_signal = None

    def combine(self, lo_signal, awg_signal):
        """Combine signal from AWG and LO."""
        ts = lo_signal["ts"]
        omega_lo = lo_signal["freq"]
        old_dim = awg_signal["inphase"].shape[0]
        new_dim = ts.shape[0]
        inphase = tf.reshape(
                    tf.image.resize(
                        tf.reshape(
                            awg_signal["inphase"],
                            shape=[1, old_dim, 1]),
                        size=[1, new_dim],
                        method='nearest'),
                    shape=[new_dim]),
        quadrature = tf.reshape(
                        tf.image.resize(
                            tf.reshape(
                                awg_signal["quadrature"],
                                shape=[1, old_dim, 1]),
                            size=[1, new_dim],
                            method='nearest'),
                        shape=[new_dim])
        self.mixed_signal = (inphase * tf.cos(omega_lo * ts)
                             + quadrature * tf.sin(omega_lo * ts))
        return self.mixed_signal


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

        self.lo_signal = {}

    def create_signal(self, control: Control):
        # TODO check somewhere that there is only 1 carrier per control
        ts = self.create_ts(control.t_start, control.t_end)
        for comp in control.comps:
            if isinstance(comp, Carrier):
                self.lo_signal["freq"] = tf.cast(comp.params["freq"],
                                                 dtype=tf.float64)
                self.lo_signal["ts"] = tf.cast(ts,
                                               dtype=tf.float64)
        return self.lo_signal


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
        # TODO move the options pwc & drag to the control object
        self.awg_signal = {}
        self.amp_tot_sq = None

# TODO create DC function

    def create_IQ(self, control: Control):
        """
        Construct the in-phase (I) and quadrature (Q) components of the signal.

        These are universal to either experiment or simulation.
        In the experiment these will be routed to AWG and mixer
        electronics, while in the simulation they provide the shapes of the
        control fields to be added to the Hamiltonian.

        """
        with tf.name_scope("I_Q_generation"):
            ts = self.create_ts(control.t_start, control.t_end)

            amp_tot_sq = 0.0
            inphase_comps = []
            quadrature_comps = []

            if (self.options == 'pwc'):
                for comp in control.comps:
                    if isinstance(comp, Envelope):
                        norm = 1
                        inphase = comp.params['inphase']
                        quadrature = comp.params['quadrature']

            elif (self.options == 'drag'):
                for comp in control.comps:
                    if isinstance(comp, Envelope):

                        amp = comp.params['amp']
                        amp_tot_sq += amp**2

                        xy_angle = comp.params['xy_angle']
                        freq_offset = comp.params['freq_offset']
                        detuning = comp.params['detuning']

                        with tf.GradientTape() as t:
                            t.watch(ts)
                            env = comp.get_shape_values(ts)

                        denv = t.gradient(env, ts)
                        phase = xy_angle - freq_offset * ts
                        inphase_comps.append(
                            amp * (
                                env * tf.cos(phase)
                                + denv/detuning * tf.sin(phase)
                            )
                        )
                        quadrature_comps.append(
                            amp * (
                                env * tf.sin(phase)
                                - denv/detuning * tf.cos(phase)
                            )
                        )
                norm = tf.sqrt(tf.cast(amp_tot_sq, tf.float64))
                inphase = tf.add_n(inphase_comps, name="Inhpase")/norm
                quadrature = tf.add_n(quadrature_comps, name="quadrature")/norm

            else:
                for comp in control.comps:
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
                inphase = tf.add_n(inphase_comps, name="Inhpase")/norm
                quadrature = tf.add_n(quadrature_comps, name="quadrature")/norm

        self.amp_tot = norm
        self.awg_signal['inphase'] = inphase
        self.awg_signal['quadrature'] = quadrature
        return self.awg_signal

    def get_I(self):
        return self.amp_tot * self.awg_signal['inphase']

    def get_Q(self):
        return self.amp_tot * self.awg_signal['quadrature']

    def plot_IQ_components(self, control: Control):
        """Plot control functions."""
        ts = self.create_ts(control.t_start, control.t_end)
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


class Generator:
    """Generator, creates signal from digital to what arrives to the chip."""

    def __init__(
            self,
            devices: dict
            ):
        # TODO consider making the dict into a list of devices
        # TODO check that you get at least 1 set of LO, AWG and mixer.
        self.devices = devices
        # TODO add line knowledge (mapping of which devices are connected)

    def generate_signals(self, controlset: ControlSet):
        # TODO deal with multiple controls within controlset
        with tf.name_scope('Signal_generation'):
            gen_signal = {}
            lo = self.devices["lo"]
            awg = self.devices["awg"]
            # TODO make mixer optional and have a signal chain (eg. Flux tuning)
            mixer = self.devices["mixer"]

            for control in controlset.controls:
                gen_signal[control.name] = {}
                lo_signal = lo.create_signal(control)
                awg_signal = awg.create_IQ(control)
                mixed_signal = mixer.combine(lo_signal, awg_signal)
                gen_signal[control.name]["ts"] = lo_signal["ts"]
                gen_signal[control.name]["values"] = mixed_signal

        self.gen_signal = gen_signal
        return gen_signal

    def plot_signals(self, controlset: ControlSet):
        for control in ControlSet:
            signal = self.gen_signal[control.name]

            """ Plotting control functions """
            plt.rcParams['figure.dpi'] = 100

            ts = signal["ts"]
            values = signal["values"]

            fig = plt.figure()
            ax = fig.add_subplot(1, 1, 1)
            ax.plot(ts.numpy(), values.numpy())
            ax.set_xlabel('Time [ns]')
            plt.title(control.name)
            plt.grid()
            plt.show(block=False)
