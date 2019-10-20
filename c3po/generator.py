"""Singal generation stack."""

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from c3po.component import C3obj, Envelope, Carrier
from c3po.controls import Control, ControlSet


class Device(C3obj):
    """Device that is part of the stack generating the control signals."""

    def __init__(
            self,
            name: str = " ",
            desc: str = " ",
            comment: str = " ",
            resolution: np.float64 = 0.0,
            t_start: np.float64 = 0.0,
            t_end: np.float64 = 0.0,
            ):
        super().__init__(
            name=name,
            desc=desc,
            comment=comment
            )
        self.resolution = resolution
        self.t_start = t_start
        self.t_end = t_end

    def prepare_plot(self):
        plt.rcParams['figure.dpi'] = 100
        fig, axs = plt.subplots(1, 1)
        self.fig = fig
        self.axs = axs
        # return self.fig, self.axs

    def calc_slice_num(self):
        res = self.resolution
        self.slice_num = int(np.abs(self.t_start - self.t_end) * res)
        # return self.slice_num

    def create_ts(
            self,
            centered: bool = True
            ):
        if self.slice_num is None:
            self.calc_slice_num()
        dt = 1/self.resolution
        if centered:
            offset = dt/2
            num = self.slice_num
        else:
            offset = 0
            num = self.slice_num + 1
        t_start = tf.constant(self.t_start + offset, dtype=tf.float64)
        t_end = tf.constant(self.t_end - offset, dtype=tf.float64)
        self.ts = tf.linspace(t_start, t_end, num)


class Mixer(Device):
    """Mixer device, combines inputs from the local oscillator and the AWG."""

    def __init__(
            self,
            name: str = " ",
            desc: str = " ",
            comment: str = " ",
            resolution: np.float64 = 0.0,
            t_start: np.float64 = 0.0,
            t_end: np.float64 = 0.0,
            ):
        super().__init__(
            name=name,
            desc=desc,
            comment=comment,
            resolution=resolution,
            t_start=t_start,
            t_end=t_end
            )

        self.output = []


    def combine(self, AWG_signal, LO_signal):
        """Combine signal from AWG and LO."""
        self.calc_slice_num()
        self.create_ts()

        ts = self.ts

        omega_lo = LO_signal["freq"]
        inphase = AWG_signal["inphase"]
        quadrature = AWG_signal["quadrature"]
        self.output = tf.zeros_like(ts)
        self.output += (inphase * tf.cos(omega_lo * ts)
                        + quadrature * tf.sin(omega_lo * ts))


class LO(Device):
    """Local oscillator device, generates a constant oscillating signal."""

    def __init__(
            self,
            name: str = " ",
            desc: str = " ",
            comment: str = " ",
            resolution: np.float64 = 0.0,
            t_start: np.float64 = 0.0,
            t_end: np.float64 = 0.0,
            ):
        super().__init__(
            name=name,
            desc=desc,
            comment=comment,
            resolution=resolution,
            t_start=t_start,
            t_end=t_end
            )

        self.LO_signal = {}

    def signal(self, controls: Control):
        for comp in controls.comps:
            if isinstance(comp, Carrier):
                self.LO_signal["freq"] = comp.params["freq"]


class AWG(Device):
    """AWG device, transforms digital input to analog signal."""

    def __init__(
            self,
            name: str = " ",
            desc: str = " ",
            comment: str = " ",
            resolution: np.float64 = 0.0,
            t_start: np.float64 = 0.0,
            t_end: np.float64 = 0.0,
            ):
        super().__init__(
            name=name,
            desc=desc,
            comment=comment,
            resolution=resolution,
            t_start=t_start,
            t_end=t_end
            )

        self.options = ""

        self.inphase = []
        self.quadrature = []
        self.amp_tot_sq = None

# TODO create DC function

    def create_IQ(self, controls: Control):
        """
        Construct the in-phase (I) and quadrature (Q) components of the signal.

        These are universal to either experiment or simulation.
        In the experiment these will be routed to AWG and mixer
        electronics, while in the simulation they provide the shapes of the
        control fields to be added to the Hamiltonian.

        """
        with tf.name_scope("I_Q_generation"):

            self.calc_slice_num()
            self.create_ts(centered=False)
            ts = self.ts

            inphase = []
            quadrature = []

            amp_tot_sq = 0.0
            inphase_comps = []
            quadrature_comps = []


            if (self.options == 'pwc'):
                for comp in controls.comps:
                    if isinstance(comp, Envelope):
                        self.amp_tot = 1
                        inphase = comp.params['inphase']
                        quadrature = comp.params['quadrature']
                        self.inphase = inphase
                        self.quadrature = quadrature

            elif (self.options == 'drag'):
                for comp in controls.comps:
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

                self.amp_tot = norm
                self.inphase = inphase
                self.quadrature = quadrature

            else:
                for comp in controls.comps:
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
                self.inphase = inphase
                self.quadrature = quadrature

    def get_I(self):
        return self.amp_tot * self.inphase

    def get_Q(self):
        return self.amp_tot * self.quadrature

    def plot_IQ_components(self):
        """Plot control functions."""
        ts = self.ts
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
            devices: dict = {},
            resolutions: dict = {}
            ):
        self.devices = devices
        self.output = None

    def generate_signals(self, controlset: ControlSet):
        with tf.name_scope('Signal_generation'):
            output = {}
            awg = self.devices["awg"]
            mixer = self.devices["mixer"]

            for controls in controlset:

                awg.t_start = ctrl.t_start
                awg.t_end = ctrl.t_end
                awg.resolutions = self.resolutions
                awg.resources = [ctrl]
                awg.resource_groups = self.resource_groups
                awg.create_IQ("awg")

                #awg.plot_IQ_components("awg")
                #awg.plot_fft_IQ_components("awg")

                mixer.t_start = ctrl.t_start
                mixer.t_end = ctrl.t_end
                mixer.resolutions = self.resolutions
                mixer.resources = [ctrl]
                mixer.resource_groups = self.resource_groups
                mixer.calc_slice_num("sim")
                mixer.create_ts("sim")

                # I = tfp.math.interp_regular_1d_grid(
                #     mixer.ts,
                #     x_ref_min = awg.ts[0],
                #     x_ref_max = awg.ts[-1],
                #     y_ref = awg.get_I()
                #     )
                # Q =  tfp.math.interp_regular_1d_grid(
                #     mixer.ts,
                #     x_ref_min = awg.ts[0],
                #     x_ref_max = awg.ts[-1],
                #     y_ref = awg.get_Q()
                #     )
                I = tf.image.resize_images(
                    awg.get_I,
                    mixer.ts.shape
                )
                Q = tf.image.resize_images(
                    awg.get_Q,
                    mixer.ts.shape
                )

                mixer.inphase = I
                mixer.quadrature = Q
                mixer.combine("sim")

                output[(ctrl.name,ctrl.get_uuid())] = {"ts" : mixer.ts}
                output[(ctrl.name,ctrl.get_uuid())].update(
                    {"signal" : mixer.output}
                    )

                self.output = output

        return output



    def plot_signals(self, resources = []):

        if resources != []:
            self.generate_signals(resources)

        for entry in self.output:
            ctrl_name = entry[0]
            control = self.output[entry]

            """ Plotting control functions """
            plt.rcParams['figure.dpi'] = 100


            ts = control["ts"]
            signal = control["signal"]

            fig = plt.figure()
            ax = fig.add_subplot(1, 1, 1)
            ax.plot(ts.numpy(), signal.numpy())
            ax.set_xlabel('Time [ns]')
            plt.title(ctrl_name)
            plt.grid()
            plt.show(block=False)


    def plot_fft_signals(self, resources = []):

        if resources != []:
            self.generate_signals(resources)

        print("WARNING: still have to adjust the x-axis")

        for entry in self.output:
            ctrl_name = entry[0]
            control = self.output[entry]


            """ Plotting control functions """
            plt.rcParams['figure.dpi'] = 100

            ts = control["ts"]
            signal = control["signal"]


            fft_signal = np.fft.fft(signal)
            fft_signal = np.fft.fftshift(fft_signal.real / max(fft_signal.real))

            plt.plot(ts, fft_signal)
            plt.title(ctrl_name + " (fft)")

            plt.show(block=False)
