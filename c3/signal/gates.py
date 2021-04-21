import hjson
import numpy as np
import tensorflow as tf
from c3.c3objs import C3obj, Quantity
from c3.signal.pulse import Envelope, Carrier
from c3.libraries.envelopes import gaussian_nonorm
import warnings
from typing import List, Dict
import copy


class Instruction:
    """
    Collection of components making up the control signal for a line.

    Parameters
    ----------
    t_start : np.float64
        Start of the signal.
    t_end : np.float64
        End of the signal.
    channels : list
        List of channel names (strings)


    Attributes
    ----------
    comps : dict
        Nested dictionary with lines and components as keys

    Example:
    comps = {
             'channel_1' : {
                            'envelope1': envelope1,
                            'envelope2': envelope2,
                            'carrier': carrier
                            }
             }

    """

    def __init__(
        self,
        name: str = " ",
        channels: List[str] = [],
        t_start: float = None,
        t_end: float = None,  # TODO remove in the long term
        # fixed_t_end: bool = True,
    ):
        self.name = name
        self.t_start = t_start
        self.t_end = t_end
        self.comps: Dict[str, Dict[str, C3obj]] = dict()
        self.__options: Dict[str, dict] = dict()
        self.fixed_t_end = True
        for chan in channels:
            self.comps[chan] = dict()
            self.__options[chan] = dict()

        self.__timings: Dict[str, tuple] = dict()
        # TODO remove redundancy of channels in instruction

    def asdict(self) -> dict:
        components = {}  # type:ignore
        for chan, item in self.comps.items():
            components[chan] = {}
            for key, comp in item.items():
                components[chan][key] = comp.asdict()
        return {"gate_length": self.t_end - self.t_start, "drive_channels": components}

    def __str__(self) -> str:
        return hjson.dumps(self.asdict())

    def add_component(self, comp: C3obj, chan: str, options=None, name=None) -> None:
        """
        Add one component, e.g. an envelope, local oscillator, to a channel.

        Parameters
        ----------
        comp : C3obj
            Component to be added.
        chan : str
            Identifier for the target channel
        options: dict
            Options for this component, available keys are
                delay: Quantity
                    Delay execution of this component by a certain time
                trigger_comp: Tuple[str]
                    Tuple of (chan, name) of component acting as trigger. Delay time will be counted beginning with end of trigger
                t_final_cut: Quantity
                    Length of component, signal will be cut after this time. Also used for the trigger. If not given this invokation from components `t_final` will be attempted.
                drag: bool
                    Use drag correction for this component.

        t_end: float
            End of this component. None will use the full instruction. If t_end is None and t_start is given a length will be inherited from the instruction.

        """
        if chan in self.comps and comp.name in self.comps[chan]:
            print(
                f"Component of instruction {self.name} has been overwritten: Channel: {chan}, Component: {comp.name}",
            )
        if name is None:
            name = comp.name
        self.comps[chan][name] = comp
        if options is None:
            options = dict()
        self.__options[chan][name] = options

    def get_optimizable_parameters(self):
        parameter_list = list()
        for chan in self.comps.keys():
            for comp in self.comps[chan]:
                for par_name, par_value in self.comps[chan][comp].params.items():
                    parameter_list.append(
                        ([self.name, chan, comp, par_name], par_value)
                    )
                for option_name, option_val in self.__options[chan][comp].items():
                    if isinstance(option_val, Quantity):
                        parameter_list.append(
                            ([self.name, chan, comp, option_name], option_val)
                        )
        return parameter_list

    def get_timings(self, chan, name, minimal_time=False):
        key = chan + "-" + name
        if key in self.__timings:
            return self.__timings[key]
        opts = self.__options[chan][name]
        comp = self.comps[chan][name]

        t_start = self.t_start
        if "delay" in opts:
            t_start += opts["delay"].get_value()
        if "trigger_comp" in opts:
            t_start += self.get_timings(*opts["trigger_comp"])[1]

        if "t_final_cut" in opts:
            t_end = t_start + opts["t_final_cut"].get_value()
        elif isinstance(comp, Envelope):
            t_end = t_start + comp.params["t_final"]
        elif minimal_time:
            t_end = t_start
        else:
            t_end = self.t_end

        if t_end > self.t_end:
            if self.fixed_t_end and not minimal_time:
                warnings.warn(
                    f"Length of instruction {self.name} is fixed, but cuts at least one component. {chan}-{name} is should end @ {t_end}, but instruction ends at {self.t_end}"
                )
                t_end = self.t_end
            elif minimal_time:
                pass
            else:
                # TODO make compatible with generator
                warnings.warn(
                    f"""T_end of {self.name} has been extended to {t_end}. This will however only take effect on the next signal generation"""
                )
                self.t_end = t_end
        self.__timings[key] = (t_start, t_end)
        return t_start, t_end

    def get_full_gate_length(self):
        t_gate_start = np.inf
        t_gate_end = -np.inf
        for chan in self.comps:
            self.__timings = dict()
            for name in self.comps[chan]:
                start, end = self.get_timings(chan, name, minimal_time=True)
                t_gate_start = min(t_gate_start, start)
                t_gate_end = max(t_gate_end, end)
                # print(end, chan, name)
        print("k")
        return t_gate_start, t_gate_end

    def auto_adjust_t_end(self, buffer=0):
        while True:
            t_end = self.get_full_gate_length()[1]
            print(t_end, self.t_end)
            if self.t_end == t_end:
                break
            self.t_end = t_end
        self.t_end = float(t_end * (1 + buffer))

    def get_awg_signal(self, chan, ts, options=None):
        amp_tot_sq = 0
        signal = tf.zeros_like(ts, tf.complex128)
        self.__timings = dict()
        for comp_name in self.comps[chan]:
            opts = copy.copy(self.__options[chan][comp_name])
            opts.update(options)
            comp = self.comps[chan][comp_name]
            t_start, t_end = self.get_timings(chan, comp_name)
            comp_ts = tf.identity(ts)
            mask = tf.ones_like(ts)
            if t_start is not None:
                comp_ts -= t_start
                mask = tf.where(ts > t_start, mask, tf.zeros_like(ts))
            if t_end is not None:
                mask = tf.where(ts < t_end, mask, tf.zeros_like(ts))
            mask_ids = tf.where(mask)
            comp_ts = tf.gather(comp_ts, indices=mask_ids)

            if isinstance(comp, Envelope):

                amp = comp.params["amp"].get_value(dtype=tf.complex128)

                amp_tot_sq += amp ** 2

                xy_angle = comp.params["xy_angle"].get_value()
                freq_offset = comp.params["freq_offset"].get_value()

                if freq_offset == 0:
                    phase = -xy_angle
                else:
                    phase = -xy_angle + freq_offset * comp_ts
                denv = None
                # TODO account for t_before
                if "drag" in opts and opts["drag"] or comp.drag:
                    dt = ts[1] - ts[0]
                    delta = -comp.params["delta"].get_value()
                    with tf.GradientTape() as t:
                        t.watch(comp_ts)
                        env = comp.get_shape_values(
                            comp_ts
                        )  # TODO t_before was ignored here
                    # Use drag_2 definition here
                    denv = t.gradient(
                        env, comp_ts, unconnected_gradients=tf.UnconnectedGradients.ZERO
                    )  # Derivative W.R.T. to bins
                    denv = denv * dt  # derivative W.R.T. to time

                    env = tf.complex(env, denv * delta)
                elif "pwc" in options and options["pwc"]:
                    inphase = comp.params["inphase"].get_value()
                    quadrature = comp.params["quadrature"].get_value()
                    tf.debugging.assert_shapes(
                        inphase,
                        quadrature,
                        message="inphase and quadrature are of different lengths.",
                    )
                    env = tf.complex(inphase, quadrature)
                    len_diff = len(ts) - len(env)
                    if len_diff > 0:
                        zeros = tf.zeros([len_diff], tf.complex128)
                        env = tf.concat([env, zeros], axis=0)
                    elif len_diff < 0:
                        print("C3 Warning: AWG has less timesteps than given PWC bins")

                else:
                    env = comp.get_shape_values(comp_ts)
                    env = tf.cast(env, tf.complex128)

                # Minus in front of the phase is equivalent of quadrature definition with -sin(phase)
                comp_sig = (
                    amp * env * tf.math.exp(tf.complex(tf.zeros_like(phase), -phase))
                )
                mask_ids = tf.reshape(mask_ids, comp_ts.shape)
                comp_sig = tf.reshape(comp_sig, comp_ts.shape[:1])

                signal += tf.scatter_nd(mask_ids, comp_sig, shape=ts.shape[:1])

        norm = tf.sqrt(tf.cast(amp_tot_sq, tf.float64))

        inphase = tf.math.real(signal)
        quadrature = tf.math.imag(signal)

        return {"inphase": inphase, "quadrature": quadrature}, norm

    def quick_setup(self, chan, qubit_freq, gate_time, v2hz=1, sideband=None) -> None:
        """
        Initialize this instruction with a default envelope and carrier.
        """
        pi_half_amp = np.pi / 2 / gate_time / v2hz * 2 * np.pi
        env_params = {
            "t_final": Quantity(value=gate_time, unit="s"),
            "amp": Quantity(
                value=pi_half_amp, min_val=0.0, max_val=3 * pi_half_amp, unit="V"
            ),
        }
        carrier_freq = qubit_freq
        if sideband:
            env_params["freq_offset"] = Quantity(value=sideband, unit="Hz 2pi")
            carrier_freq -= sideband
        self.comps[chan]["gaussian"] = Envelope(
            "gaussian", shape=gaussian_nonorm, params=env_params
        )
        self.comps[chan]["carrier"] = Carrier(
            "Carr_" + chan, params={"freq": Quantity(value=carrier_freq, unit="Hz 2pi")}
        )
