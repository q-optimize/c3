import types
import json
import copy
import tensorflow as tf
import numpy as np
from c3.signal.pulse import Envelope, Carrier
from c3.c3objs import Quantity, C3obj
import matplotlib.pyplot as plt


class Device(C3obj):
    """A Device that is part of the stack generating the instruction signals.

    Parameters
    ----------
    resolution: np.float64
        Number of samples per second this device operates at.
    """

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

    def write_config(self):
        """
        Return the current device as a JSON compatible dict.
        """
        cfg = copy.deepcopy(self.__dict__)
        cfg.pop('signal', None)
        cfg.pop('ts', None)
        cfg.pop('amp_tot', None)
        cfg.pop('amp_tot_sq', None)
        cfg.pop('freq', None)
        for p in cfg['params']:
            cfg['params'][p] = float(cfg['params'][p])
        return cfg

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
        """
        Effective number of time slices given start, end and resolution.

        Parameters
        ----------
        t_start: np.float64
            Starting time for this device.
        t_end: np.float64
            End time for this device.
        """
        res = self.resolution
        self.slice_num = int(np.abs(t_start - t_end) * res)
        # return self.slice_num

    def create_ts(
        self,
        t_start: np.float64,
        t_end: np.float64,
        centered: bool = True
    ):
        """
        Compute time samples.

        Parameters
        ----------
        t_start: np.float64
            Starting time for this device.
        t_end: np.float64
            End time for this device.
        centered: boolean
            Sample in the middle of an interval, otherwise at the beginning.
        """
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


class Readout(Device):
    """Mimic the readout process by multiplying a state phase with a factor and offset.

    Parameters
    ----------
    factor: Quantity
    offset: Quantity
    """

    def __init__(
            self,
            name: str = "readout",
            desc: str = " ",
            comment: str = " ",
            resolution: np.float64 = 0.0,
            factor: Quantity = None,
            offset: Quantity = None,
    ):
        super().__init__(
            name=name,
            desc=desc,
            comment=comment,
            resolution=resolution
        )
        self.signal = None
        self.params['factor'] = factor
        self.params['offset'] = offset

    def readout(self, phase):
        """
        Apply the readout rescaling

        Parameters
        ----------
        phase: tf.float64
            Raw phase of a quantum state

        Returns
        -------
        tf.float64
            Rescaled readout value
        """
        offset = self.params['offset'].get_value()
        factor = self.params['factor'].get_value()
        return phase * factor + offset


class Volts_to_Hertz(Device):
    """Convert the voltage signal to an amplitude to plug into the model Hamiltonian.

    Parameters
    ----------
    V_to_Hz: Quantity
        Conversion factor.
    offset: tf.float64
        Drive frequency offset.
    """

    def __init__(
            self,
            name: str = "v_to_hz",
            desc: str = " ",
            comment: str = " ",
            resolution: np.float64 = 0.0,
            V_to_Hz: Quantity = None
    ):
        super().__init__(
            name=name,
            desc=desc,
            comment=comment,
            resolution=resolution
        )
        self.signal = None
        self.params['V_to_Hz'] = V_to_Hz

    def transform(self, mixed_signal, drive_frequency):
        """Transform signal from value of V to Hz.

        Parameters
        ----------
        mixed_signal: tf.Tensor
            Waveform as line voltages after IQ mixing
        drive_frequency: Quantity
            For frequency-dependent attenuation

        Returns
        -------
        tf.Tensor
            Waveform as control amplitudes
        """
        v2hz = self.params['V_to_Hz'].get_value()
        self.signal = mixed_signal * v2hz
        return self.signal


class Digital_to_Analog(Device):
    """Take the values at the awg resolution to the simulation resolution."""

    def __init__(
            self,
            name: str = "dac",
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

        self.signal = {}
        self.ts = None

    def resample(self, awg_signal, t_start, t_end):
        """Resample the awg values to higher resolution.

        Parameters
        ----------
        awg_signal: tf.Tensor
            Bandwith-limited, low-resolution AWG signal.
        t_start: np.float64
            Beginning of the signal.
        t_end: np.float64
            End of the signal.

        Returns
        -------
        dict
            Inphase and Quadrature compontent of the upsampled signal.
        """
        ts = self.create_ts(t_start, t_end, centered=True)
        old_dim = awg_signal["inphase"].shape[0]
        new_dim = ts.shape[0]
        inphase = tf.reshape(
            tf.image.resize(
                tf.reshape(awg_signal["inphase"], shape=[1, old_dim, 1]), size=[1, new_dim], method='nearest'
            ),
            shape=[new_dim]
        )
        quadrature = tf.reshape(
            tf.image.resize(
                tf.reshape(awg_signal["quadrature"], shape=[1, old_dim, 1]), size=[1, new_dim], method='nearest'
            ),
            shape=[new_dim]
        )
        self.ts = ts
        self.signal['inphase'] = inphase
        self.signal['quadrature'] = quadrature
        return {"inphase": inphase, "quadrature": quadrature}


class Filter(Device):
    # TODO This can apply a general function to a signal.
    """Apply a filter function to the signal."""

    def __init__(
            self,
            name: str = "FLTR",
            desc: str = " ",
            comment: str = " ",
            resolution: np.float64 = 0.0,
            filter_function: types.FunctionType = None,
    ):
        super().__init__(
            name=name,
            desc=desc,
            comment=comment,
            resolution=resolution
        )
        self.filter_function = filter_function
        self.signal = None

    def filter(self, Hz_signal):
        """Apply a filter function to the signal."""
        self.signal = self.filter_function(Hz_signal)
        return self.signal


class FluxTuning(Device):
    """
    Flux tunable qubit frequency.

    Parameters
    ----------
    phi_0 : Quantity
        Flux bias.
    Phi : Quantity
        Current flux.
    omega_0 : Quantity
        Maximum frequency.

    """

    def __init__(
            self,
            name: str = " ",
            desc: str = " ",
            comment: str = " ",
            resolution: np.float64 = 0.0,
            phi_0: np.float = 0.0,
            phi: np.float = 0.0,
            omega_0: np.float = 0.0,
            d = None
    ):

        super().__init__(
            name=name,
            desc=desc,
            comment=comment,
            resolution=resolution
        )
        self.params['phi_0'] = phi_0
        self.params['phi'] = phi
        self.params['omega_0'] = omega_0
        if d:
            self.params['d'] = d
        self.freq = None

    def frequency(self, signal):
        """
        Compute the qubit frequency resulting from an applied flux.

        Parameters
        ----------
        signal : tf.float64


        Returns
        -------
        tf.float64
            Qubit frequency.
        """
        pi = tf.constant(np.pi, dtype=tf.float64)
        phi = self.params['phi'].get_value()
        omega_0 = self.params['omega_0'].get_value()
        phi_0 = self.params['phi_0'].get_value()

        if 'd' in self.params:
            d = self.params['d'].get_value()
#             print('assuming asymmetric transmon with d=', d)
            base_freq = omega_0 * tf.sqrt(tf.sqrt(
                tf.cos(pi * phi / phi_0)**2 + d**2 * tf.sin(pi * phi / phi_0)**2
            ))
            self.freq = omega_0 * tf.sqrt(tf.sqrt(
                tf.cos(pi * (phi + signal) / phi_0)**2 + d**2 * tf.sin(pi * (phi + signal) / phi_0)**2
            )) - base_freq
        else:
            base_freq = omega_0 * tf.sqrt(tf.abs(tf.cos(pi * phi / phi_0)))
            self.freq = omega_0 * tf.sqrt(tf.abs(tf.cos(pi * (phi + signal) / phi_0))) - base_freq
        return self.freq


class FluxTuningLinear(Device):
    """
    Flux tunable qubit frequency linear adjustment.

    Parameters
    ----------
    phi_0 : Quantity
        Flux bias.
    Phi : Quantity
        Current flux.
    omega_0 : Quantity
        Maximum frequency.

    """

    def __init__(
            self,
            name: str = " ",
            desc: str = " ",
            comment: str = " ",
            resolution: np.float64 = 0.0,
            phi_0: np.float = 0.0,
            phi: np.float = 0.0,
            omega_0: np.float = 0.0,
            d = None
    ):

        super().__init__(
            name=name,
            desc=desc,
            comment=comment,
            resolution=resolution
        )
        self.params['phi_0'] = phi_0
        self.params['phi'] = phi
        self.params['omega_0'] = omega_0
        if d:
            self.params['d'] = d
        self.freq = None

    def frequency(self, signal):
        """
        Compute the qubit frequency resulting from an applied flux.

        Parameters
        ----------
        signal : tf.float64


        Returns
        -------
        tf.float64
            Qubit frequency.
        """
        pi = tf.constant(np.pi, dtype=tf.float64)
        phi = self.params['phi'].get_value()
        omega_0 = self.params['omega_0'].get_value()
        phi_0 = self.params['phi_0'].get_value()
        if 'd' in self.params:
            d = self.params['d'].get_value()
            max_freq = omega_0               
            min_freq = omega_0 * tf.sqrt(tf.sqrt(
                tf.cos(pi * 0.5)**2 + d**2 * tf.sin(pi * 0.5)**2
            ))
        else:
            max_freq = omega_0
            min_freq = tf.constant(0.0, dtype=tf.float64)
        self.freq = 2 * (signal-phi) * (min_freq - max_freq)
        return self.freq

class Response(Device):
    """Make the AWG signal physical by convolution with a Gaussian to limit bandwith.

    Parameters
    ----------
    rise_time : Quantity
        Time constant for the gaussian convolution.
    """

    def __init__(
            self,
            name: str = "resp",
            desc: str = " ",
            comment: str = " ",
            resolution: np.float64 = 0.0,
            rise_time: Quantity = None,
    ):
        super().__init__(
            name=name,
            desc=desc,
            comment=comment,
            resolution=resolution
        )
        self.params['rise_time'] = rise_time
        self.signal = None

    def convolve(self, signal: list, resp_shape: list):
        """
        Compute the convolution with a function.

        Parameters
        ----------
        signal : list
            Potentially unlimited signal samples.
        resp_shape : list
            Samples of the function to model limited bandwidth.

        Returns
        -------
        tf.Tensor
            Processed signal.

        """
        convolution = tf.zeros(0, dtype=tf.float64)
        signal = tf.concat(
            [tf.zeros(len(resp_shape), dtype=tf.float64), signal, tf.zeros(len(resp_shape), dtype=tf.float64)], 0
        )
        for p in range(len(signal) - 2 * len(resp_shape)):
            convolution = tf.concat(
                [
                    convolution,
                    tf.reshape(
                        tf.math.reduce_sum(tf.math.multiply(signal[p:p + len(resp_shape)], resp_shape)), shape=[1]
                    )
                ], 0
            )
        return convolution

    def process(self, iq_signal):
        """
        Apply a Gaussian shaped limiting function to an IQ signal.

        Parameters
        ----------
        iq_signal : dict
            I and Q components of an AWG signal.

        Returns
        -------
        dict
            Bandwidth limited IQ signal.

        """
        n_ts = tf.floor(self.params['rise_time'].get_value() * self.resolution)
        ts = tf.linspace(
            tf.constant(0.0, dtype=tf.float64),
            self.params['rise_time'].get_value(),
            tf.cast(n_ts, tf.int32)
        )
        cen = tf.cast(
            (self.params['rise_time'].get_value() - 1 / self.resolution) / 2,
            tf.float64
        )
        sigma = self.params['rise_time'].get_value() / 4
        gauss = tf.exp(-(ts - cen) ** 2 / (2 * sigma * sigma))
        offset = tf.exp(-(-1 - cen) ** 2 / (2 * sigma * sigma))
        # TODO make sure ratio of risetime and resolution is an integer
        risefun = gauss - offset
        inphase = self.convolve(iq_signal['inphase'], risefun / tf.reduce_sum(risefun))
        quadrature = self.convolve(iq_signal['quadrature'], risefun / tf.reduce_sum(risefun))
        self.signal = {'inphase': inphase, 'quadrature': quadrature}
        return self.signal


class Mixer(Device):
    """Mixer device, combines inputs from the local oscillator and the AWG."""

    def __init__(
            self,
            name: str = "mixer",
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
        self.signal = None

    def combine(self, lo_signal, awg_signal):
        """Combine signal from AWG and LO.

        Parameters
        ----------
        lo_signal : dict
            Local oscillator signal.
        awg_signal : dict
            Waveform generator signal.

        Returns
        -------
        dict
            Mixed signal.
        """
        cos, sin = lo_signal["values"]
        inphase = awg_signal["inphase"]
        quadrature = awg_signal["quadrature"]
        self.signal = (inphase * cos + quadrature * sin)
        #TODO: check if signs are right
        return self.signal


class LONoise(Device):
    """Noise applied to the local oscillator"""

    def __init__(
            self,
            name: str = "lo_noise",
            desc: str = " ",
            comment: str = " ",
            resolution: np.float64 = 0.0,
            noise_perc: Quantity = None
    ):
        super().__init__(
            name=name,
            desc=desc,
            comment=comment,
            resolution=resolution
        )
        self.signal = None
        self.params['noise_perc'] = noise_perc

    def distort(self, lo_signal):
        """Distort signal by adding noise."""
        noise_perc = self.params['noise_perc'].get_value()
        cos, sin = lo_signal["values"]
        cos = cos + noise_perc * np.random.normal(loc=0.0,scale=1.0, size=len(cos))
        sin = sin + noise_perc * np.random.normal(loc=0.0,scale=1.0, size=len(sin))
        lo_signal["values"] = (cos, sin)
        self.signal = lo_signal
        return self.signal

class Pink_Noise(Device):
    """Device creating pink noise, i.e. 1/f noise."""

    def __init__(
            self,
            name: str = "pink_noise",
            desc: str = " ",
            comment: str = " ",
            resolution: np.float64 = 0.0,
            noise_strength: Quantity = None,
            bfl_num: Quantity = None,
#             infrared_cutoff = None,
#             ultraviolet_cutoff = None
    ):
        super().__init__(
            name=name,
            desc=desc,
            comment=comment,
            resolution=resolution
        )
        self.signal = None
        self.params['noise_strength'] = noise_strength
        if not bfl_num:
            bfl_num = Quantity(
                value=5,
                min=1,
                max=10
            )
        self.params['bfl_num'] = bfl_num
#         self.params['infrared_cutoff'] = infrared_cutoff
#         self.params['ultraviolet_cutoff'] = ultraviolet_cutoff
        self.ts = None
        self.signal = None

    def distort(self, mixed_signal):
        bfl_num = np.int(self.params['bfl_num'].get_value().numpy())
        noise_strength = self.params['noise_strength'].get_value().numpy()

#         noise = []
#         bfls = np.random.randint(2, size=bfl_num)
#         for step in range(len(mixed_signal)):
#             for indx in range(bfl_num):
#                 if np.floor(np.random.random() * (10^indx))==0:
#                     bfls[indx] = not(bfls[indx])
#             noise.append(np.sum(bfls) * noise_strength)

        noise = []
        bfls = 2 * np.random.randint(2, size=bfl_num) - 1
        num_steps = len(mixed_signal)
        flip_rates = np.logspace(0, np.log(num_steps), num=bfl_num+1, endpoint=True, base=10.0)
        for step in range(num_steps):
            for indx in range(bfl_num):
                if np.floor(np.random.random() * flip_rates[indx+1])==0:
                    bfls[indx] = - bfls[indx]
            noise.append(np.sum(bfls) * noise_strength)

        self.noise = noise
        self.signal = mixed_signal + tf.constant(noise, shape=mixed_signal.shape, dtype= tf.float64)
        return self.signal



class LO(Device):
    """Local oscillator device, generates a constant oscillating signal."""

    def __init__(
        self,
        name: str = "lo",
        desc: str = " ",
        comment: str = " ",
        resolution: np.float64 = 0.0,
        phase_noise: Quantity = None,
        amp_noise: Quantity = None,
        freq_noise: Quantity = None
    ):
        super().__init__(
            name=name,
            desc=desc,
            comment=comment,
            resolution=resolution
        )
        self.phase_noise = phase_noise
        self.freq_noise = freq_noise
        self.amp_noise = amp_noise
        self.signal = {}

    def create_signal(self, channel: dict, t_start: float, t_end: float):
        """
        Generate a sinusodial signal.

        Parameters
        ----------
        channel : dict
            Drive channels.
        t_start : float
            Beginning of the signal.
        t_end : float
            End of the signal.

        Returns
        -------
        dict, tf.float64
            Local oscillator signal and frequency.

        """
        # TODO check somewhere that there is only 1 carrier per instruction
        ts = self.create_ts(t_start, t_end, centered=True)
        dt = ts[1] - ts[0]
        phase_noise = self.phase_noise
        amp_noise = self.amp_noise
        freq_noise = self.freq_noise
        for c in channel:
            comp = channel[c]
            if isinstance(comp, Carrier):
                cos, sin = [], []
                omega_lo = comp.params['freq'].get_value()
                if amp_noise and freq_noise:
                    print('amp and freq noise')
                    phi = omega_lo * ts[0]
                    for t in ts:
                        A = np.random.normal(loc=1.0, scale=amp_noise)
                        cos.append(A * np.cos(phi))
                        sin.append(A * np.sin(phi))
                        omega = omega_lo + np.random.normal(loc=0.0,scale=freq_noise)
                        phi = phi + omega * dt
                elif amp_noise and phase_noise:
                    print('amp and phase noise')
                    for t in ts:
                        A = np.random.normal(loc=1.0,scale=amp_noise)
                        phi = np.random.normal(loc=0.0,scale=phase_noise)
                        cos.append(A * np.cos(omega_lo * t + phi))
                        sin.append(A * np.sin(omega_lo * t + phi))
                elif amp_noise:
                    print('amp noise')
                    for t in ts:
                        A = np.random.normal(loc=1.0,scale=amp_noise)
                        cos.append(A * np.cos(omega_lo * t))
                        sin.append(A * np.sin(omega_lo * t))
                elif phase_noise:
                    print('phase noise')
                    for t in ts:
                        phi = np.random.normal(loc=0.0,scale=phase_noise)
                        cos.append(np.cos(omega_lo * t + phi))
                        sin.append(np.sin(omega_lo * t + phi))
                elif freq_noise:
                    phi = omega_lo * ts[0]
                    for t in ts:
                        cos.append(np.cos(phi))
                        sin.append(np.sin(phi))
                        omega = omega_lo + np.random.normal(loc=0.0,scale=freq_noise)
                        phi = phi + omega * dt
                else:
                    cos = tf.cos(omega_lo * ts)
                    sin = tf.sin(omega_lo * ts)
                self.signal["values"] = (cos, sin)
                self.signal["ts"] = ts
                return self.signal, omega_lo


# TODO real AWG has 16bits plus noise
class AWG(Device):
    """AWG device, transforms digital input to analog signal.

    Parameters
    ----------
    logdir : str
        Filepath to store generated waveforms.
    """

    def __init__(
        self,
        name: str = "awg",
        desc: str = " ",
        comment: str = " ",
        resolution: np.float64 = 0.0,
        logdir: str = '/tmp/'
    ):
        super().__init__(
            name=name,
            desc=desc,
            comment=comment,
            resolution=resolution
        )

        self.__options = ""
        self.logdir = logdir
        self.logname = "awg.log"
        # TODO move the options pwc & drag to the instruction object
        self.signal = {}
        self.amp_tot_sq = None

# TODO create DC function

    # TODO make AWG take offset from the previous point
    def create_IQ(self, channel: str, components: dict, t_start: float, t_end: float):
        """
        Construct the in-phase (I) and quadrature (Q) components of the signal.
        These are universal to either experiment or simulation.
        In the xperiment these will be routed to AWG and mixer
        electronics, while in the simulation they provide the shapes of the
        instruction fields to be added to the Hamiltonian.

        Parameters
        ----------
        channel : str
            Identifier for the selected drive line.
        components : dict
            Separate signals to be combined onto this drive line.
        t_start : float
            Beginning of the signal.
        t_end : float
            End of the signal.

        Returns
        -------
        dict
            Waveforms as I and Q components.

        """
        with tf.name_scope("I_Q_generation"):
            self.signal[channel] = {}
            ts = self.create_ts(t_start, t_end, centered=True)
            self.ts = ts
            dt = ts[1] - ts[0]
            t_before = ts[0] - dt
            amp_tot_sq = 0.0
            inphase_comps = []
            quadrature_comps = []

            if (self.__options == 'pwc'):
                #TODO add buffer for pwc case
                amp_tot_sq = 0
                for key in components:
                    comp = components[key]
                    if isinstance(comp, Envelope):
                        amp_tot_sq += 1
                        inphase = comp.params['inphase'].get_value()
                        quadrature = comp.params['quadrature'].get_value()
                        xy_angle = comp.params['xy_angle'].get_value()
                        phase = xy_angle

                        if len(inphase) != len(quadrature):
                            raise ValueError('inphase and quadrature are of different lengths.')
                        if len(inphase) < len(ts):
                            zeros = tf.constant(
                               np.zeros(len(ts)-len(inphase)),
                               dtype=inphase.dtype
                            )
                            inphase = tf.concat([inphase, zeros], axis=0)
                            quadrature = tf.concat([quadrature, zeros], axis=0)
                        inphase_comps.append(
                            inphase * tf.cos(phase)
                            + quadrature * tf.sin(phase)
                        )
                        quadrature_comps.append(
                            - inphase * tf.sin(phase)
                            + quadrature * tf.cos(phase)

                        )

                norm = tf.sqrt(tf.cast(amp_tot_sq, tf.float64))
                inphase = tf.add_n(inphase_comps, name="inphase")
                quadrature = tf.add_n(quadrature_comps, name="quadrature")
                freq_offset = 0.0

            elif (self.__options == 'drag') or (self.__options == 'drag_2'):
                for key in components:
                    comp = components[key]
                    if isinstance(comp, Envelope):

                        amp = comp.params['amp'].get_value()
                        amp_tot_sq += amp**2

                        xy_angle = comp.params['xy_angle'].get_value()
                        freq_offset = comp.params['freq_offset'].get_value()
                        # TODO should we remove this redefinition?
                        delta = - comp.params['delta'].get_value()
                        if (self.__options == 'drag_2'):
                            delta = delta * dt

                        with tf.GradientTape() as t:
                            t.watch(ts)
                            env = comp.get_shape_values(ts)
                            # TODO option to have t_before = 0
                            # env = comp.get_shape_values(ts, t_before)

                        denv = t.gradient(env, ts)
                        if denv is None:
                            denv = tf.zeros_like(ts, dtype=tf.float64)
                        #TODO cehck again the sign here
                        phase = - xy_angle + freq_offset * ts
                        inphase_comps.append(
                            amp * (
                                env * tf.cos(phase)
                                + denv * delta * tf.sin(phase)
                            )
                        )
                        quadrature_comps.append(
                            amp * (
                                - env * tf.sin(phase)
                                + denv * delta * tf.cos(phase)
                            )
                        )
                norm = tf.sqrt(tf.cast(amp_tot_sq, tf.float64))
                inphase = tf.add_n(inphase_comps, name="inphase")
                quadrature = tf.add_n(quadrature_comps, name="quadrature")

            else:
                for key in components:
                    comp = components[key]
                    if isinstance(comp, Envelope):

                        amp = comp.params['amp'].get_value()

                        amp_tot_sq += amp**2

                        xy_angle = comp.params['xy_angle'].get_value()
                        freq_offset = comp.params['freq_offset'].get_value()
                        # TODO: check again the sign in front of offset
                        phase = - xy_angle + freq_offset * ts
                        inphase_comps.append(
                            amp * comp.get_shape_values(ts) * tf.cos(phase)
                        )
                        quadrature_comps.append(
                            - amp * comp.get_shape_values(ts) * tf.sin(phase)
                        )

                norm = tf.sqrt(tf.cast(amp_tot_sq, tf.float64))
                inphase = tf.add_n(inphase_comps, name="inphase")
                quadrature = tf.add_n(quadrature_comps, name="quadrature")

        self.amp_tot = norm
        self.signal[channel]['inphase'] = inphase
        self.signal[channel]['quadrature'] = quadrature
        # self.log_shapes()
        return {"inphase": inphase, "quadrature": quadrature}
        # TODO decide when and where to return/store params scaled or not

    def get_average_amp(self, line):
        """
        Compute average and sum of the amplitudes. Used to estimate effective drive power for non-trivial shapes.

        Returns
        -------
        tuple
            Average and sum.
        """
        In = self.get_I(line)
        Qu = self.get_Q(line)
        amp_per_bin = tf.sqrt(tf.abs(In)**2 + tf.abs(Qu)**2)
        av = tf.reduce_mean(amp_per_bin)
        sum = tf.reduce_sum(amp_per_bin)
        return av, sum

    def get_I(self, line):
        return self.signal[line]['inphase'] # * self.amp_tot

    def get_Q(self, line):
        return self.signal[line]['quadrature'] # * self.amp_tot

    def enable_drag(self):
        self.__options = 'drag'

    def enable_drag_2(self):
        self.__options = 'drag_2'
        
    def enable_pwc(self):
        self.__options = 'pwc'

    def log_shapes(self):
        # TODO log shapes in the generator instead
        with open(self.logdir + self.logname, 'a') as logfile:
            signal = {}
            for key in self.signal:
                signal[key] = self.signal[key].numpy().tolist()
            logfile.write(json.dumps(signal))
            logfile.write("\n")
            logfile.flush()
