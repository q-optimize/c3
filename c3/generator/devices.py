import os
import hjson
import tempfile
import tensorflow as tf
import numpy as np
from c3.signal.pulse import Envelope, Carrier
from c3.signal.gates import Instruction
from c3.c3objs import Quantity, C3obj
from c3.utils.tf_utils import tf_convolve

devices = dict()


def dev_reg_deco(func):
    """
    Decorator for making registry of functions
    """
    devices[str(func.__name__)] = func
    return func


class Device(C3obj):
    """A Device that is part of the stack generating the instruction signals.

    Parameters
    ----------
    resolution: np.float64
        Number of samples per second this device operates at.
    """

    def __init__(self, **props):
        if "inputs" not in self.__dict__:
            self.inputs = props.pop("inputs", 0)
        if "outputs" not in self.__dict__:
            self.outputs = props.pop("outputs", 0)
        self.resolution = props.pop("resolution", 0)
        name = props.pop("name")
        desc = props.pop("desc", "")
        comment = props.pop("comment", "")

        # Because of legacy usage, we might have parameters given withing props iself
        # or in a "params" field. Here we combine them.
        params = props.pop("params", {})
        params.update(props)
        super().__init__(name, desc, comment, params)
        self.signal = {}

    def write_config(self, filepath: str) -> None:
        """
        Write dictionary to a HJSON file.
        """
        with open(filepath, "w") as cfg_file:
            hjson.dump(self.asdict(), cfg_file)

    def asdict(self) -> dict:
        params = {}
        for key, item in self.params.items():
            params[key] = item.asdict()
        return {
            "c3type": self.__class__.__name__,
            "inputs": self.inputs,
            "outputs": self.outputs,
            "params": params,
            "resolution": self.resolution,
        }

    def __str__(self) -> str:
        return hjson.dumps(self.asdict())

    def calc_slice_num(self, t_start: np.float64, t_end: np.float64):
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

    def create_ts(self, t_start: np.float64, t_end: np.float64, centered: bool = True):
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
        if not hasattr(self, "slice_num"):
            self.calc_slice_num(t_start, t_end)
        dt = 1 / self.resolution
        # TODO This type of centering does not guarantee zeros at the ends
        if centered:
            offset = dt / 2
            num = self.slice_num
        else:
            offset = 0
            num = self.slice_num + 1
        t_start = tf.constant(t_start + offset, dtype=tf.float64)
        t_end = tf.constant(t_end - offset, dtype=tf.float64)
        # TODO: adjust the way we calculate the time slices for devices
        # ts = tf.range(t_start, t_end + 1e-16, dt)
        ts = tf.linspace(t_start, t_end, num)
        return ts


@dev_reg_deco
class Readout(Device):
    """Mimic the readout process by multiplying a state phase with a factor and offset.

    Parameters
    ----------
    factor: Quantity
    offset: Quantity
    """

    def __init__(self, **props):
        super().__init__(**props)
        if "factor" not in self.params:
            raise Exception("C3:ERROR: Readout device needs a 'factor' parameter.")
        if "offset" not in self.params:
            raise Exception("C3:ERROR: Readout device needs an 'offset' parameter.")

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
        offset = self.params["offset"].get_value()
        factor = self.params["factor"].get_value()
        return phase * factor + offset


@dev_reg_deco
class VoltsToHertz(Device):
    """Convert the voltage signal to an amplitude to plug into the model Hamiltonian.

    Parameters
    ----------
    V_to_Hz: Quantity
        Conversion factor.
    offset: tf.float64
        Drive frequency offset.
    """

    def __init__(self, **props):
        super().__init__(**props)
        self.inputs = props.pop("inputs", 1)
        self.outputs = props.pop("outputs", 1)

    def process(self, instr, chan, mixed_signal):
        """Transform signal from value of V to Hz.

        Parameters
        ----------
        mixed_signal: tf.Tensor
            Waveform as line voltages after IQ mixing

        Returns
        -------
        tf.Tensor
            Waveform as control amplitudes
        """
        v2hz = self.params["V_to_Hz"].get_value()
        self.signal["values"] = mixed_signal["values"] * v2hz
        self.signal["ts"] = mixed_signal["ts"]
        return self.signal


@dev_reg_deco
class DigitalToAnalog(Device):
    """Take the values at the awg resolution to the simulation resolution."""

    def __init__(self, **props):
        super().__init__(**props)
        self.inputs = props.pop("inputs", 1)
        self.outputs = props.pop("outputs", 1)
        self.ts = None
        self.sampling_method = props.pop("sampling_method", "nearest")

    def process(self, instr, chan, awg_signal):
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
        ts = self.create_ts(instr.t_start, instr.t_end, centered=True)
        old_dim = awg_signal["inphase"].shape[0]
        new_dim = ts.shape[0]
        # TODO add following zeros
        inphase = tf.reshape(
            tf.image.resize(
                tf.reshape(awg_signal["inphase"], shape=[1, old_dim, 1]),
                size=[1, new_dim],
                method=self.sampling_method,
            ),
            shape=[new_dim],
        )
        inphase = tf.cast(inphase, dtype=tf.float64)
        quadrature = tf.reshape(
            tf.image.resize(
                tf.reshape(awg_signal["quadrature"], shape=[1, old_dim, 1]),
                size=[1, new_dim],
                method=self.sampling_method,
            ),
            shape=[new_dim],
        )
        quadrature = tf.cast(quadrature, dtype=tf.float64)
        self.signal["ts"] = ts
        self.signal["inphase"] = inphase
        self.signal["quadrature"] = quadrature
        return self.signal


@dev_reg_deco
class Filter(Device):
    # TODO This can apply a general function to a signal.
    """Apply a filter function to the signal."""

    def __init__(self, **props):
        raise Exception("C3:ERROR Not yet implemented.")
        self.filter_function = props["filter_function"]
        super().__init__(**props)

    def process(self, instr, chan, Hz_signal):
        """Apply a filter function to the signal."""
        self.signal = self.filter_function(Hz_signal)
        return self.signal


@dev_reg_deco
class FluxTuning(Device):
    """
    Flux tunable qubit frequency.

    Parameters
    ----------
    phi_0 : Quantity
        Flux bias.
    phi : Quantity
        Current flux.
    omega_0 : Quantity
        Maximum frequency.

    """

    def __init__(self, **props):
        super().__init__(**props)
        self.inputs = props.pop("inputs", 1)
        self.outputs = props.pop("outputs", 1)
        for par in ["phi_0", "phi", "omega_0", "anhar"]:
            if par not in self.params:
                raise Exception(
                    f"C3:ERROR: {self.__class__}  needs a '{par}' parameter."
                )

    def get_factor(self, phi):
        pi = tf.constant(np.pi, dtype=tf.float64)
        phi_0 = tf.cast(self.params["phi_0"].get_value(), tf.float64)

        if "d" in self.params:
            d = self.params["d"].get_value()
            factor = tf.sqrt(
                tf.sqrt(
                    tf.cos(pi * phi / phi_0) ** 2
                    + d ** 2 * tf.sin(pi * phi / phi_0) ** 2
                )
            )
        else:
            factor = tf.sqrt(tf.abs(tf.cos(pi * phi / phi_0)))
        return factor

    def get_freq(self, phi):
        # TODO: Check how the time dependency affects the frequency. (Koch et al. , 2007)
        omega_0 = self.params["omega_0"].get_value()
        anhar = self.params["anhar"].get_value()
        biased_freq = (omega_0 - anhar) * self.get_factor(phi) + anhar
        return biased_freq

    def process(self, instr: Instruction, chan: str, signal_in):
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
        phi = self.params["phi"].get_value()
        signal = signal_in["values"]
        self.signal["ts"] = signal_in["ts"]
        freq = self.get_freq(phi + signal) - self.get_freq(phi)
        self.signal["values"] = freq
        return self.signal


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

    def __init__(self, **props):
        super().__init__(**props)
        for par in ["phi_0", "Phi", "omega_0"]:
            if par not in self.params:
                raise Exception(
                    f"C3:ERROR: {self.__class__}  needs a '{par}' parameter."
                )
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
        phi = self.params["phi"].get_value()
        omega_0 = self.params["omega_0"].get_value()
        # phi_0 = self.params["phi_0"].get_value()
        if "d" in self.params:
            d = self.params["d"].get_value()
            max_freq = omega_0
            min_freq = omega_0 * tf.sqrt(
                tf.sqrt(tf.cos(pi * 0.5) ** 2 + d ** 2 * tf.sin(pi * 0.5) ** 2)
            )
        else:
            max_freq = omega_0
            min_freq = tf.constant(0.0, dtype=tf.float64)
        self.freq = 2 * (signal - phi) * (min_freq - max_freq)
        return self.freq


@dev_reg_deco
class Response(Device):
    """Make the AWG signal physical by convolution with a Gaussian to limit bandwith.

    Parameters
    ----------
    rise_time : Quantity
        Time constant for the gaussian convolution.
    """

    def __init__(self, **props):
        super().__init__(**props)
        self.inputs = props.pop("inputs", 1)
        self.outputs = props.pop("outputs", 1)

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
            [
                tf.zeros(len(resp_shape), dtype=tf.float64),
                signal,
                tf.zeros(len(resp_shape), dtype=tf.float64),
            ],
            0,
        )
        for p in range(len(signal) - 2 * len(resp_shape)):
            convolution = tf.concat(
                [
                    convolution,
                    tf.reshape(
                        tf.math.reduce_sum(
                            tf.math.multiply(
                                signal[p : p + len(resp_shape)], resp_shape
                            )
                        ),
                        shape=[1],
                    ),
                ],
                0,
            )
        return convolution

    def process(self, instr, chan, iq_signal):
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
        n_ts = tf.floor(self.params["rise_time"].get_value() * self.resolution)
        ts = tf.linspace(
            tf.constant(0.0, dtype=tf.float64),
            self.params["rise_time"].get_value(),
            tf.cast(n_ts, tf.int32),
        )
        cen = tf.cast(
            (self.params["rise_time"].get_value() - 1 / self.resolution) / 2, tf.float64
        )
        sigma = self.params["rise_time"].get_value() / 4
        gauss = tf.exp(-((ts - cen) ** 2) / (2 * sigma * sigma))
        offset = tf.exp(-((-1 - cen) ** 2) / (2 * sigma * sigma))
        # TODO make sure ratio of risetime and resolution is an integer
        risefun = gauss - offset
        inphase = self.convolve(iq_signal["inphase"], risefun / tf.reduce_sum(risefun))
        quadrature = self.convolve(
            iq_signal["quadrature"], risefun / tf.reduce_sum(risefun)
        )
        self.signal = {
            "inphase": inphase,
            "quadrature": quadrature,
            "ts": self.create_ts(instr.t_start, instr.t_end, centered=True),
        }
        return self.signal


@dev_reg_deco
class ResponseFFT(Device):
    """Make the AWG signal physical by convolution with a Gaussian to limit bandwith.

    Parameters
    ----------
    rise_time : Quantity
        Time constant for the gaussian convolution.
    """

    def __init__(self, **props):
        super().__init__(**props)
        self.inputs = props.pop("inputs", 1)
        self.outputs = props.pop("outputs", 1)

    def process(self, instr, chan, iq_signal):
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
        # print(tf.abs(1 / tf.math.reduce_mean(iq_signal['ts'][1] - iq_signal['ts'][0]) ),self.resolution)
        assert (
            tf.abs((iq_signal["ts"][1] - iq_signal["ts"][0]) - 1 / self.resolution)
            < 1e-15
        ) or True
        n_ts = tf.floor(self.params["rise_time"].get_value() * self.resolution)
        ts = tf.linspace(
            tf.constant(0.0, dtype=tf.float64),
            self.params["rise_time"].get_value(),
            tf.cast(n_ts, tf.int32),
        )
        cen = tf.cast(
            (self.params["rise_time"].get_value() - 1 / self.resolution) / 2, tf.float64
        )
        sigma = self.params["rise_time"].get_value() / 4
        gauss = tf.exp(-((ts - cen) ** 2) / (2 * sigma * sigma))
        offset = tf.exp(-((-1 - cen) ** 2) / (2 * sigma * sigma))

        risefun = gauss - offset
        inphase = tf_convolve(iq_signal["inphase"], risefun / tf.reduce_sum(risefun))
        quadrature = tf_convolve(
            iq_signal["quadrature"], risefun / tf.reduce_sum(risefun)
        )

        inphase = tf.math.real(inphase)
        quadrature = tf.math.real(quadrature)
        self.signal = {
            "inphase": inphase,
            "quadrature": quadrature,
            "ts": iq_signal["ts"],
        }
        return self.signal


@dev_reg_deco
class StepFuncFilter(Device):
    """
    Base class for filters that are based on the step response function
    Step function has to be defined explicetly
    """

    def __init__(self, **props):
        super().__init__(**props)
        self.inputs = props.pop("inputs", 1)
        self.outputs = props.pop("outputs", 1)

    def step_response_function(self, ts):
        raise NotImplementedError()

    def process(self, instr, chan, signal_in):
        ts = tf.identity(signal_in["ts"])
        step_response = self.step_response_function(ts)
        step_response = tf.concat([[0], step_response], axis=0)
        impulse_response = step_response[1:] - step_response[:-1]
        signal_out = dict()
        for key, signal in signal_in.items():
            if key == "ts":
                continue
            signal_out[key] = tf.cast(tf_convolve(signal, impulse_response), tf.float64)
        signal_out["ts"] = signal_in["ts"]

        return signal_out


@dev_reg_deco
class ExponentialIIR(StepFuncFilter):
    """Implement IIR filter with step response of the form
    s(t) = (1 + A * exp(-t / t_iir) )

    Parameters
    ----------
    time_iir: Quantity
        Time constant for the filtering.
    amp: Quantity

    """

    def step_response_function(self, ts):
        time_iir = self.params["time_iir"]
        amp = self.params["amp"]
        step_response = 1 + amp * tf.exp(-ts / time_iir)
        return step_response


@dev_reg_deco
class HighpassExponential(StepFuncFilter):
    """Implement Highpass filter based on exponential with step response of the form
    s(t) = exp(-t / t_hp)

    Parameters
    ----------
    time_iir: Quantity
        Time constant for the filtering.
    amp: Quantity

    """

    def step_response_function(self, ts):
        time_hp = self.params["time_hp"]
        return tf.exp(-ts / time_hp)


@dev_reg_deco
class SkinEffectResponse(StepFuncFilter):
    """Implement Highpass filter based on exponential with step response of the form
    s(t) = exp(-t / t_hp)

    Parameters
    ----------
    time_iir: Quantity
        Time constant for the filtering.
    amp: Quantity

    """

    def step_response_function(self, ts):
        alpha = self.params["alpha"]
        return tf.math.erfc(alpha / 21 / tf.math.sqrt(np.abs(ts)))


class HighpassFilter(Device):
    """Introduce a highpass filter

    Parameters
    ----------
    cutoff : Quantity
        cutoff frequency of highpass filter
    keep_mean : bool
        should the mean of the signal be restored
    """

    def __init__(self, **props):
        super().__init__(**props)
        self.inputs = props.pop("inputs", 1)
        self.outputs = props.pop("outputs", 1)
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
            [
                tf.zeros(int(len(resp_shape) // 2), dtype=tf.float64),
                signal,
                tf.zeros(int(len(resp_shape) * 3 / 2) + 1, dtype=tf.float64),
            ],
            0,
        )
        for p in range(len(signal) - 2 * len(resp_shape)):
            convolution = tf.concat(
                [
                    convolution,
                    tf.reshape(
                        tf.math.reduce_sum(
                            tf.math.multiply(
                                signal[p : p + len(resp_shape)], resp_shape
                            )
                        ),
                        shape=[1],
                    ),
                ],
                0,
            )
        return convolution

    def process(self, instr, chan, iq_signal):
        """
        Apply a highpass cutoff to an IQ signal.

        Parameters
        ----------
        iq_signal : dict
            I and Q components of an AWG signal.

        Returns
        -------
        dict
            Filtered IQ signal.

        """
        fc = self.params["cutoff"].get_value() / self.resolution

        if self.params["rise_time"]:
            tb = self.params["rise_time"].get_value() / self.resolution
        else:
            tb = fc / 2

        # fc = 1e7 / self.resolution
        # tb = fc / 2

        N_ts = tf.cast(tf.math.ceil(4 / tb), dtype=tf.int32)
        N_ts += 1 - tf.math.mod(N_ts, 2)  # make n_ts odd
        if N_ts > len(iq_signal["inphase"] * 100):
            self.signal = iq_signal
            return self.signal

        pi = tf.cast(np.pi, dtype=tf.double)

        n = tf.cast(tf.range(N_ts), dtype=tf.double)

        x = 2 * fc * (n - (N_ts - 1) / 2)
        h = tf.sin(pi * x) / (pi * x)
        h = tf.where(tf.math.is_nan(h), tf.ones_like(h), h)
        w = tf.signal.hamming_window(N_ts)
        w = tf.cast(w, dtype=tf.double)
        h *= w
        h /= -tf.reduce_sum(h)
        h = tf.where(tf.cast(n, dtype=tf.int32) == (N_ts - 1) // 2, tf.ones_like(h), h)
        inphase = self.convolve(iq_signal["inphase"], h)
        quadrature = self.convolve(iq_signal["quadrature"], h)
        self.signal = {
            "inphase": inphase,
            "quadrature": quadrature,
            "ts": iq_signal["ts"],
        }
        return self.signal


@dev_reg_deco
class Mixer(Device):
    """Mixer device, combines inputs from the local oscillator and the AWG."""

    def __init__(self, **props):
        super().__init__(**props)
        self.inputs = props.pop("inputs", 2)
        self.outputs = props.pop("outputs", 1)

    def process(self, instr: Instruction, chan: str, in1: dict, in2: dict):
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
        i1 = in1["inphase"]
        q1 = in1["quadrature"]
        i2 = in2["inphase"]
        q2 = in2["quadrature"]
        self.signal = {"values": i1 * i2 + q1 * q2, "ts": in1["ts"]}
        # See Engineer's Guide Eq. 88
        # TODO: Check consistency of the signs between Mixer, LO and AWG classes
        return self.signal


@dev_reg_deco
class LONoise(Device):
    """Noise applied to the local oscillator"""

    def __init__(self, **props):
        super().__init__(**props)
        self.inputs = props.pop("inputs", 1)
        self.outputs = props.pop("outputs", 1)
        self.signal = None
        self.params["noise_perc"] = props.pop("noise_perc")

    def process(self, instr, chan, lo_signal):
        """Distort signal by adding noise."""
        noise_perc = self.params["noise_perc"].get_value()
        cos, sin = lo_signal["values"]
        cos = cos + noise_perc * np.random.normal(loc=0.0, scale=1.0, size=len(cos))
        sin = sin + noise_perc * np.random.normal(loc=0.0, scale=1.0, size=len(sin))
        lo_signal["values"] = (cos, sin)
        self.signal = lo_signal
        return self.signal


@dev_reg_deco
class Additive_Noise(Device):
    """Noise applied to a signal"""

    def __init__(self, **props):
        super().__init__(**props)
        self.inputs = props.pop("inputs", 1)
        self.outputs = props.pop("outputs", 1)
        self.signal = None
        self.params["noise_amp"] = props.pop("noise_amp")

    def get_noise(self, sig):
        noise_amp = self.params["noise_amp"].get_value()
        return noise_amp * np.random.normal(size=tf.shape(sig), loc=0.0, scale=1.0)

    def process(self, instr, chan, signal):
        """Distort signal by adding noise."""
        noise_amp = self.params["noise_amp"].get_value()
        out_signal = {"ts": signal["ts"]}
        for k, sig in signal.items():
            if k != "ts" and "noise" not in k:
                if noise_amp < 1e-17:
                    noise = tf.zeros_like(sig)
                else:
                    noise = tf.constant(
                        self.get_noise(sig), shape=sig.shape, dtype=tf.float64
                    )
                noise_key = "noise" + ("-" + k if k != "values" else "")
                out_signal[noise_key] = noise

                out_signal[k] = sig + noise
        self.signal = out_signal
        return self.signal


@dev_reg_deco
class DC_Noise(Additive_Noise):
    """Add a random constant offset to the signals"""

    def get_noise(self, sig):
        noise_amp = self.params["noise_amp"].get_value()
        return tf.ones_like(sig) * tf.constant(
            noise_amp * np.random.normal(loc=0.0, scale=1.0)
        )


@dev_reg_deco
class Pink_Noise(Additive_Noise):
    """Device creating pink noise, i.e. 1/f noise."""

    def __init__(self, **props):
        super().__init__(**props)
        self.params["bfl_num"] = props.pop(
            "bfl_num", Quantity(value=5, min_val=1, max_val=10)
        )

    def get_noise(self, sig):
        noise_amp = self.params["noise_amp"].get_value().numpy()
        bfl_num = np.int(self.params["bfl_num"].get_value().numpy())
        noise = []
        bfls = 2 * np.random.randint(2, size=bfl_num) - 1
        num_steps = len(sig)
        flip_rates = np.logspace(
            0, np.log(num_steps), num=bfl_num + 1, endpoint=True, base=10.0
        )
        for step in range(num_steps):
            for indx in range(bfl_num):
                if np.floor(np.random.random() * flip_rates[indx + 1]) == 0:
                    bfls[indx] = -bfls[indx]
            noise.append(np.sum(bfls) * noise_amp)
        return noise


@dev_reg_deco
class DC_Offset(Device):
    """Noise applied to a signal"""

    def __init__(self, **props):
        super().__init__(**props)
        self.inputs = props.pop("inputs", 1)
        self.outputs = props.pop("outputs", 1)
        self.signal = None
        self.params["offset_amp"] = props.pop("offset_amp")

    def process(self, instr, chan, signal):
        """Distort signal by adding noise."""
        offset_amp = self.params["offset_amp"].get_value()
        if np.abs(offset_amp) < 1e-17:
            self.signal = signal
            return signal
        out_signal = {}
        if type(signal) is dict:
            for k, sig in signal.items():
                out_signal[k] = sig + offset_amp
        else:
            out_signal = signal + offset_amp
        self.signal = out_signal
        return self.signal


@dev_reg_deco
class LO(Device):
    """Local oscillator device, generates a constant oscillating signal."""

    def __init__(self, **props):
        super().__init__(**props)
        self.outputs = props.pop("outputs", 1)
        self.phase_noise = props.pop("phase_noise", 0)
        self.freq_noise = props.pop("freq_noise", 0)
        self.amp_noise = props.pop("amp_noise", 0)

    def process(self, instr: Instruction, chan: str) -> dict:
        # TODO check somewhere that there is only 1 carrier per instruction
        ts = self.create_ts(instr.t_start, instr.t_end, centered=True)
        dt = ts[1] - ts[0]
        phase_noise = self.phase_noise
        amp_noise = self.amp_noise
        freq_noise = self.freq_noise
        components = instr.comps
        for comp in components[chan].values():
            if isinstance(comp, Carrier):
                cos, sin = [], []
                omega_lo = comp.params["freq"].get_value()
                if amp_noise and freq_noise:
                    print("amp and freq noise")
                    phi = omega_lo * ts[0]
                    for t in ts:
                        A = np.random.normal(loc=1.0, scale=amp_noise)
                        cos.append(A * np.cos(phi))
                        sin.append(A * np.sin(phi))
                        omega = omega_lo + np.random.normal(loc=0.0, scale=freq_noise)
                        phi = phi + omega * dt
                elif amp_noise and phase_noise:
                    print("amp and phase noise")
                    for t in ts:
                        A = np.random.normal(loc=1.0, scale=amp_noise)
                        phi = np.random.normal(loc=0.0, scale=phase_noise)
                        cos.append(A * np.cos(omega_lo * t + phi))
                        sin.append(A * np.sin(omega_lo * t + phi))
                elif amp_noise:
                    print("amp noise")
                    for t in ts:
                        A = np.random.normal(loc=1.0, scale=amp_noise)
                        cos.append(A * np.cos(omega_lo * t))
                        sin.append(A * np.sin(omega_lo * t))
                elif phase_noise:
                    print("phase noise")
                    for t in ts:
                        phi = np.random.normal(loc=0.0, scale=phase_noise)
                        cos.append(np.cos(omega_lo * t + phi))
                        sin.append(np.sin(omega_lo * t + phi))
                elif freq_noise:
                    phi = omega_lo * ts[0]
                    for t in ts:
                        cos.append(np.cos(phi))
                        sin.append(np.sin(phi))
                        omega = omega_lo + np.random.normal(loc=0.0, scale=freq_noise)
                        phi = phi + omega * dt
                else:
                    cos = tf.cos(omega_lo * ts)
                    sin = tf.sin(omega_lo * ts)
                self.signal["inphase"] = cos
                self.signal["quadrature"] = sin
                self.signal["ts"] = ts
        return self.signal


# TODO real AWG has 16bits plus noise
@dev_reg_deco
class AWG(Device):
    """AWG device, transforms digital input to analog signal.

    Parameters
    ----------
    logdir : str
        Filepath to store generated waveforms.
    """

    def __init__(self, **props):
        self.__options = ""
        self.logdir = props.pop(
            "logdir", os.path.join(tempfile.gettempdir(), "c3logs", "AWG")
        )
        self.logname = "awg.log"
        options = props.pop("options", "")
        super().__init__(**props)
        self.outputs = props.pop("outputs", 1)
        # TODO move the options pwc & drag to the instruction object
        self.amp_tot_sq = None
        self.process = self.create_IQ
        if options == "drag":
            self.enable_drag()
        elif options == "drag_2":
            self.enable_drag_2()
        self.centered_ts = True

    def asdict(self) -> dict:
        awg_dict = super().asdict()
        awg_dict["options"] = self.__options
        return awg_dict

    # TODO create DC function

    # TODO make AWG take offset from the previous point
    def create_IQ(self, instr: Instruction, chan: str) -> dict:
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
        ts = self.create_ts(instr.t_start, instr.t_end, centered=True)
        components = instr.comps
        self.ts = ts
        # dt = ts[1] - ts[0]
        # t_before = ts[0] - dt
        amp_tot_sq = 0.0
        inphase_comps = []
        quadrature_comps = []

        for comp in components[chan].values():
            if isinstance(comp, Envelope):

                amp = comp.params["amp"].get_value()

                amp_tot_sq += amp ** 2

                xy_angle = comp.params["xy_angle"].get_value()
                freq_offset = comp.params["freq_offset"].get_value()
                phase = -xy_angle + freq_offset * ts
                env = comp.get_shape_values(ts)
                # TODO option to have t_before
                # env = comp.get_shape_values(ts, t_before)
                inphase_comps.append(amp * env * tf.cos(phase))
                quadrature_comps.append(-amp * env * tf.sin(phase))

        norm = tf.sqrt(tf.cast(amp_tot_sq, tf.float64))
        if len(inphase_comps) > 1:
            inphase = tf.add_n(inphase_comps, name="inphase")
            quadrature = tf.add_n(quadrature_comps, name="quadrature")
        else:
            inphase = inphase_comps[0]
            quadrature = quadrature_comps[0]

        self.amp_tot = norm
        self.signal[chan] = {"inphase": inphase, "quadrature": quadrature, "ts": ts}
        return {"inphase": inphase, "quadrature": quadrature, "ts": ts}

    def create_IQ_drag(self, instr: Instruction, chan: str) -> dict:
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
        ts = self.create_ts(instr.t_start, instr.t_end, centered=True)
        components = instr.comps
        self.ts = ts
        dt = ts[1] - ts[0]
        t_before = ts[0] - dt
        amp_tot_sq = 0.0
        inphase_comps = []
        quadrature_comps = []

        for comp in components[chan].values():
            if isinstance(comp, Envelope):

                amp = comp.params["amp"].get_value()
                amp_tot_sq += amp ** 2

                xy_angle = comp.params["xy_angle"].get_value()
                freq_offset = comp.params["freq_offset"].get_value()
                # TODO should we remove this redefinition?
                delta = -comp.params["delta"].get_value()
                if self.__options == "drag_2":
                    delta = delta * dt

                with tf.GradientTape() as t:
                    t.watch(ts)
                    env = comp.get_shape_values(ts, t_before)
                    # TODO option to have t_before = 0
                    # env = comp.get_shape_values(ts, t_before)

                denv = t.gradient(env, ts)
                if denv is None:
                    denv = tf.zeros_like(ts, dtype=tf.float64)
                phase = -xy_angle + freq_offset * ts
                inphase_comps.append(
                    amp * (env * tf.cos(phase) + denv * delta * tf.sin(phase))
                )
                quadrature_comps.append(
                    amp * (denv * delta * tf.cos(phase) - env * tf.sin(phase))
                )
        norm = tf.sqrt(tf.cast(amp_tot_sq, tf.float64))
        inphase = tf.add_n(inphase_comps, name="inphase")
        quadrature = tf.add_n(quadrature_comps, name="quadrature")

        self.amp_tot = norm
        self.signal[chan] = {"inphase": inphase, "quadrature": quadrature, "ts": ts}
        return {"inphase": inphase, "quadrature": quadrature, "ts": ts}

    def create_IQ_pwc(self, instr: Instruction, chan: str) -> dict:
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
        ts = self.create_ts(instr.t_start, instr.t_end, centered=True)
        components = instr.comps
        self.ts = ts
        # dt = ts[1] - ts[0]
        amp_tot_sq = 0.0
        inphase_comps = []
        quadrature_comps = []

        amp_tot_sq = 0
        for comp in components[chan].values():
            if isinstance(comp, Envelope):
                amp_tot_sq += 1
                if comp.shape is None:
                    inphase = comp.params["inphase"].get_value()
                    quadrature = comp.params["quadrature"].get_value()
                else:
                    shape = comp.get_shape_values(ts)
                    inphase = tf.math.real(shape)
                    quadrature = tf.math.imag(shape)
                xy_angle = comp.params["xy_angle"].get_value()
                freq_offset = comp.params["freq_offset"].get_value()
                phase = -xy_angle + freq_offset * ts

                if len(inphase) != len(quadrature):
                    raise ValueError("inphase and quadrature are of different lengths.")
                elif len(inphase) < len(ts):
                    zeros = tf.constant(
                        np.zeros(len(ts) - len(inphase)), dtype=inphase.dtype
                    )
                    inphase = tf.concat([inphase, zeros], axis=0)
                    quadrature = tf.concat([quadrature, zeros], axis=0)

                inphase_comps.append(
                    inphase * tf.cos(phase) + quadrature * tf.sin(phase)
                )
                quadrature_comps.append(
                    quadrature * tf.cos(phase) - inphase * tf.sin(phase)
                )

        norm = tf.sqrt(tf.cast(amp_tot_sq, tf.float64))
        inphase = tf.add_n(inphase_comps, name="inphase")
        quadrature = tf.add_n(quadrature_comps, name="quadrature")

        self.amp_tot = norm
        self.signal[chan] = {"inphase": inphase, "quadrature": quadrature, "ts": ts}
        return {"inphase": inphase, "quadrature": quadrature, "ts": ts}

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
        amp_per_bin = tf.sqrt(tf.abs(In) ** 2 + tf.abs(Qu) ** 2)
        av = tf.reduce_mean(amp_per_bin)
        sum = tf.reduce_sum(amp_per_bin)
        return av, sum

    def get_I(self, line):
        return self.signal[line]["inphase"]  # * self.amp_tot

    def get_Q(self, line):
        return self.signal[line]["quadrature"]  # * self.amp_tot

    def enable_drag(self):
        self.process = self.create_IQ_drag

    def enable_drag_2(self):
        self.process = self.create_IQ_drag
        self.__options = "drag_2"

    def enable_pwc(self):
        self.process = self.create_IQ_pwc

    def log_shapes(self):
        # TODO log shapes in the generator instead
        with open(self.logdir + self.logname, "a") as logfile:
            signal = {}
            for key in self.signal:
                signal[key] = self.signal[key].numpy().tolist()
            logfile.write(hjson.dumps(signal))
            logfile.write("\n")
            logfile.flush()
