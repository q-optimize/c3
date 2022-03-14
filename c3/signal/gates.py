import hjson
import numpy as np
import tensorflow as tf
from c3.c3objs import C3obj, Quantity, hjson_encode
from c3.signal.pulse import Envelope, Carrier
from c3.libraries.envelopes import gaussian_nonorm
import warnings
from typing import List, Dict, Any
import copy
from c3.libraries.constants import GATES
from c3.utils.qt_utils import np_kron_n, insert_mat_kron
from c3.utils.tf_utils import tf_project_to_comp
from c3.signal.pulse import components as comp_lib


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
        targets: list = None,
        params: dict = None,
        ideal: np.ndarray = None,
        channels: List[str] = [],
        t_start: float = 0.0,
        t_end: float = 0.0,
        # fixed_t_end: bool = True,
    ):
        self.name = name
        self.targets = targets
        self.params: dict = {}
        if isinstance(params, dict):
            self.params.update(params)
        if t_start is not None:
            warnings.warn(
                "t_start will be removed in the future. Do not set it anymore.",
                category=DeprecationWarning,
            )
        self.t_start = t_start
        self.t_end = t_end
        self.comps: Dict[str, Dict[str, C3obj]] = dict()
        self._options: Dict[str, dict] = dict()
        self.fixed_t_end = True
        if ideal is not None:
            self.ideal = ideal
        else:
            gate_list = []
            # legacy use
            for key in name.split(":"):
                if key in GATES:
                    gate_list.append(GATES[key])
                else:
                    warnings.warn(f"No ideal gate found for gate: {key}")
            self.ideal = np_kron_n(gate_list)
        for chan in channels:
            self.comps[chan] = dict()
            self._options[chan] = dict()

        self._timings: Dict[str, tuple] = dict()

    def as_openqasm(self) -> dict:
        asdict: Dict[str, Any] = {
            "name": self.name,
            "qubits": self.targets,
            "params": self.params,
        }
        if self.ideal:
            asdict["ideal"] = self.ideal
        return asdict

    def get_ideal_gate(self, dims, index=None):
        if self.ideal is None:
            raise Exception(
                "C3:ERROR: No ideal representation definded for gate"
                f" {self.get_key()}"
            )

        targets = self.targets
        if targets is None:
            targets = list(range(len(dims)))

        ideal_gate = insert_mat_kron(
            [2] * len(dims),  # we compare to the computational basis
            targets,
            [self.ideal],
        )

        if index:
            ideal_gate = tf_project_to_comp(
                ideal_gate, dims=[2] * len(dims), index=index
            )

        return ideal_gate

    def get_key(self) -> str:
        if self.targets is None:
            return self.name
        return self.name + str(self.targets)

    def asdict(self) -> dict:
        components = {}  # type:ignore
        for chan, item in self.comps.items():
            components[chan] = {}
            for key, comp in item.items():
                components[chan][key] = comp.asdict()
        out_dict = copy.deepcopy(self.__dict__)
        out_dict["ideal"] = out_dict["ideal"]
        out_dict.pop("_timings")
        out_dict.pop("t_start")
        out_dict.pop("t_end")
        out_dict["gate_length"] = self.t_end - self.t_start
        out_dict["drive_channels"] = out_dict.pop("comps")
        return out_dict

    def from_dict(self, cfg, name=None):
        self.__init__(
            name=cfg["name"] if "name" in cfg else name,
            targets=cfg["targets"] if "targets" in cfg else None,
            params=cfg["params"] if "params" in cfg else None,
            ideal=np.array(cfg["ideal"]) if "ideal" in cfg else None,
            channels=cfg["drive_channels"].keys(),
            t_start=0.0,
            t_end=cfg["gate_length"],
        )

        options = cfg.pop("_options", None)
        components = cfg.pop("drive_channels")
        self.__dict__.update(cfg)
        for drive_chan, comps in components.items():
            for comp, props in comps.items():
                ctype = props.pop("c3type")
                if "name" not in props:
                    props["name"] = comp
                self.add_component(
                    comp_lib[ctype](**props),
                    chan=drive_chan,
                    options=options[drive_chan][comp] if options else None,
                    name=comp,
                )

    def __repr__(self):
        return f"Instruction[{self.get_key()}]"

    def __str__(self) -> str:
        return hjson.dumps(self.asdict(), default=hjson_encode)

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
                f"Component of instruction {self.get_key()} has been overwritten: Channel: {chan}, Component: {comp.name}",
            )
        if name is None:
            name = comp.name
        self.comps[chan][name] = comp
        if options is None:
            options = dict()
        for k, v in options.items():
            if isinstance(v, dict):
                options[k] = Quantity(**v)
        self._options[chan][name] = options

    def get_optimizable_parameters(self):
        parameter_list = list()
        for chan in self.comps.keys():
            for comp in self.comps[chan]:
                for par_name, par_value in self.comps[chan][comp].params.items():
                    parameter_list.append(
                        ([self.get_key(), chan, comp, par_name], par_value)
                    )
                for option_name, option_val in self._options[chan][comp].items():
                    if isinstance(option_val, Quantity):
                        parameter_list.append(
                            (
                                [
                                    self.get_key(),
                                    chan,
                                    comp,
                                    option_name,
                                ],
                                option_val,
                            )
                        )
        return parameter_list

    def get_timings(self, chan, name, minimal_time=False):
        key = chan + "-" + name
        if key in self._timings:
            return self._timings[key]
        opts = self._options[chan][name]
        comp = self.comps[chan][name]

        t_start = self.t_start
        if "delay" in opts:
            t_start += opts["delay"].get_value()
        if "trigger_comp" in opts:
            t_start += self.get_timings(*opts["trigger_comp"])[1]

        if "t_final_cut" in opts:
            t_end = t_start + opts["t_final_cut"].get_value()
        elif isinstance(comp, Envelope):
            t_end = t_start + comp.params["t_final"].get_value()
        elif minimal_time:
            t_end = t_start
        else:
            t_end = self.t_end

        # TODO: The following needs to go. We need proper configuration or an error.
        # if t_end > self.t_end:
        #     if self.fixed_t_end and not minimal_time:
        #         warnings.warn(
        #             f"Length of instruction {self.get_key()} is fixed, but cuts at least one component. {chan}-{name} is should end @ {t_end}, but instruction ends at {self.t_end}"
        #         )
        #         t_end = self.t_end
        #     elif minimal_time:
        #         pass
        #     else:
        #         # TODO make compatible with generator
        #         warnings.warn(
        #             f"""T_end of {self.get_key()} has been extended to {t_end}. This will however only take effect on the next signal generation"""
        #         )
        #         self.t_end = t_end
        self._timings[key] = (t_start, t_end)
        return t_start, t_end

    def get_full_gate_length(self):
        t_gate_start = np.inf
        t_gate_end = -np.inf
        for chan in self.comps:
            self._timings = dict()
            for name in self.comps[chan]:
                start, end = self.get_timings(chan, name, minimal_time=True)
                t_gate_start = min(t_gate_start, start)
                t_gate_end = max(t_gate_end, end)
        return t_gate_start, t_gate_end

    def auto_adjust_t_end(self, buffer=0):
        while True:
            t_end = self.get_full_gate_length()[1]
            if self.t_end == t_end:
                break
            self.t_end = t_end
        self.t_end = float(t_end * (1 + buffer))

    def get_awg_signal(self, chan, ts, options=None):
        amp_tot_sq = 0
        signal = tf.zeros_like(ts, tf.complex128)
        self._timings = dict()
        dt = ts[1] - ts[0]
        for comp_name in self.comps[chan]:
            opts = copy.copy(self._options[chan][comp_name])
            opts.update(options)
            comp = self.comps[chan][comp_name]
            t_start, t_end = self.get_timings(chan, comp_name)
            ts_off = ts - t_start
            if isinstance(comp, Envelope):

                amp_re = comp.params["amp"].get_value()
                amp = tf.complex(amp_re, tf.zeros_like(amp_re))

                amp_tot_sq += amp**2

                xy_angle = comp.params["xy_angle"].get_value()
                freq_offset = comp.params["freq_offset"].get_value()
                phase = -xy_angle - freq_offset * ts_off
                denv = None
                if comp.drag or opts.pop("drag", False) or opts.pop("drag_2", False):

                    delta = comp.params["delta"].get_value()
                    with tf.GradientTape() as t:
                        t.watch(ts_off)
                        env = comp.get_shape_values(ts_off, t_end - t_start)
                    denv = t.gradient(
                        env, ts_off, unconnected_gradients=tf.UnconnectedGradients.ZERO
                    )  # Derivative W.R.T. to bins
                    if not opts.pop("drag", False):
                        # Use drag_2 definition here
                        denv = denv * dt  # derivative W.R.T. time

                    env = tf.complex(env, -denv * delta)
                elif "pwc" in options and options["pwc"]:
                    inphase = comp.params["inphase"].get_value()
                    quadrature = comp.params["quadrature"].get_value()
                    if not inphase.shape == quadrature.shape:
                        raise Exception(
                            "C3:Error:inphase and quadrature are of different lengths."
                        )
                    env = tf.complex(inphase, quadrature)
                    len_diff = len(ts) - len(env)
                    if len_diff > 0:
                        zeros = tf.zeros([len_diff], tf.complex128)
                        env = tf.concat([env, zeros], axis=0)
                    elif len_diff < 0:
                        print("C3 Warning: AWG has less timesteps than given PWC bins")

                else:
                    env = comp.get_shape_values(ts_off, t_end - t_start)
                    env = tf.cast(env, tf.complex128)

                signal += (
                    amp * env * tf.math.exp(tf.complex(tf.zeros_like(phase), phase))
                )

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
        self.add_component(
            comp=Envelope(
                "gaussian", shape=gaussian_nonorm, params=env_params, use_t_before=True
            ),
            chan=chan,
        )

        self.add_component(
            comp=Carrier(
                "carrier", params={"freq": Quantity(value=carrier_freq, unit="Hz 2pi")}
            ),
            chan=chan,
        )
