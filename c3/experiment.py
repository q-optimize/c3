"""
Experiment class that models and simulates the whole experiment.

It combines the information about the model of the quantum device, the control stack and the operations that can be
done on the device.

Given this information an experiment run is simulated, returning either processes, states or populations.
"""

import os
import json
import pickle
import itertools
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from c3.utils import tf_utils


# TODO add case where one only wants to pass a list of quantity objects?
class Experiment:
    """
    It models all of the behaviour of the physical experiment, serving as a
    host for the individual parts making up the experiment.

    Parameters
    ----------
    model: Model
        The underlying physical device.
    generator: Generator
        The infrastructure for generating and sending control signals to the
        device.
    gateset: GateSet
        A gate level description of the operations implemented by control
        pulses.

    """

    def __init__(self, pmap=None):
        self.pmap = pmap

        self.unitaries = {}
        self.dUs = {}

    def write_config(self):
        """
        Return the current experiment as a JSON compatible dict.

        EXPERIMENTAL
        """
        cfg = {}
        cfg['model'] = model.write_config()
        cfg['generator'] = generator.write_config()
        cfg['gateset'] = self.pmap.write_config()
        return cfg

    def set_created_by(self, config):
        """
        Store the config file location used to created this experiment.
        """
        self.created_by = config
    
    def evaluate(self, seqs):
        """
        Compute the population values for a given sequence of operations.

        Parameters
        ----------
        seqs: str list
            A list of control pulses/gates to perform on the device.

        Returns
        -------
        list
            A list of populations

        """
        model = self.pmap.model
        Us = tf_utils.evaluate_sequences(self.unitaries, seqs)
        psi_init = model.tasks["init_ground"].initialise(
            model.drift_H,
            model.lindbladian
        )
        self.psi_init = psi_init
        populations = []
        for U in Us:
            psi_final = tf.matmul(U, self.psi_init)
            pops = self.populations(
                psi_final, model.lindbladian
            )
            populations.append(pops)
        return populations

    def process(self, populations,  labels=None):
        """
        Apply a readout procedure to a population vector. Very specialized
        at the moment.

        Parameters
        ----------
        populations: list
            List of populations from evaluating.

        labels: list
            List of state labels specifying a subspace.

        Returns
        -------
        list
            A list of processed populations.

        """
        model = self.pmap.model
        populations_final = []
        for pops in populations:
            # TODO: Loop over all tasks in a general fashion
            # TODO: Selecting states by label in the case of computational space
            if "conf_matrix" in model.tasks:
                pops = model.tasks["conf_matrix"].confuse(pops)
                if labels is not None:
                    pops_select = 0
                    for label in labels:
                        pops_select += pops[
                            model.comp_state_labels.index(label)
                        ]
                    pops = pops_select
                else:
                    pops = tf.reshape(pops, [pops.shape[0]])
            else:
                if labels is not None:
                    pops_select = 0
                    for label in labels:
                        try:
                            pops_select += pops[
                                model.state_labels.index(label)
                            ]
                        except ValueError:
                            raise Exception(
                                f"C3:ERROR:State {label} not defined. Available are:\n"
                                f"{model.state_labels}"
                            )
                    pops = pops_select
                else:
                    pops = tf.reshape(pops, [pops.shape[0]])
            if "meas_rescale" in model.tasks:
                pops = model.tasks["meas_rescale"].rescale(pops)
            populations_final.append(pops)
        return populations_final

    def get_gates(self):
        """
        Compute the unitary representation of operations. If no operations are
        specified in self.__opt_gates the complete gateset is computed.

        Returns
        -------
        dict
            A dictionary of gate names and their unitary representation.
        """
        model = self.pmap.model
        generator = self.pmap.generator
        instructions = self.pmap.instructions
        gates = {}
        if "__opt_gates" in self.__dict__:
            gate_keys = self.__opt_gates
        else:
            gate_keys = instructions.keys()
        for gate in gate_keys:
            try:
                instr = instructions[gate]
            except KeyError:
                raise Exception(f"C3:Error: Gate \'{gate}\' is not defined."
                                f" Available gates are:\n {list(instructions.keys())}.")
            signal, ts = generator.generate_signals(instr)
            U = self.propagation(signal, ts, gate)
            if model.use_FR:
                # TODO change LO freq to at the level of a line
                freqs = {}
                framechanges = {}
                for line, ctrls in instr.comps.items():
                    # TODO calculate properly the average frequency that each qubit sees
                    offset = 0.0
                    if "gauss" in ctrls:
                        if ctrls['gauss'].params["amp"] != 0.0:
                            offset = ctrls['gauss'].params['freq_offset'].get_value()
                    if "flux" in ctrls:
                        if ctrls['flux'].params["amp"] != 0.0:
                            offset = ctrls['flux'].params['freq_offset'].get_value()
                    if "pwc" in ctrls:
                        offset = ctrls['pwc'].params['freq_offset'].get_value()
                    # print("gate: ", gate, "; line: ", line, "; offset: ", offset)
                    freqs[line] = tf.cast(
                        ctrls['carrier'].params['freq'].get_value()
                        + offset,
                        tf.complex128
                    )
                    framechanges[line] = tf.cast(
                        ctrls['carrier'].params['framechange'].get_value(),
                        tf.complex128
                    )
                t_final = tf.constant(
                    instr.t_end - instr.t_start,
                    dtype=tf.complex128
                )
                FR = model.get_Frame_Rotation(
                    t_final,
                    freqs,
                    framechanges
                )
                if model.lindbladian:
                    SFR = tf_utils.tf_super(FR)
                    U = tf.matmul(SFR, U)
                    self.FR = SFR
                else:
                    U = tf.matmul(FR, U)
                    self.FR = FR
            if model.dephasing_strength != 0.0:
                if not model.lindbladian:
                    raise ValueError(
                        'Dephasing can only be added when lindblad is on.'
                    )
                else:
                    amps = {}
                    for line, ctrls in instr.comps.items():
                        amp, sum = generator.devices['awg'].get_average_amp()
                        amps[line] = tf.cast(amp, tf.complex128)
                    t_final = tf.constant(
                        instr.t_end - instr.t_start,
                        dtype=tf.complex128
                    )
                    dephasing_channel = model.get_dephasing_channel(
                        t_final,
                        amps
                    )
                    U = tf.matmul(dephasing_channel, U)
            gates[gate] = U
            self.unitaries = gates
        return gates

    def propagation(
        self,
        signal: dict,
        ts,
        gate
    ):
        """
        Solve the equation of motion (Lindblad or Schrödinger) for a given control signal and Hamiltonians.

        Parameters
        ----------
        signal: dict
            Waveform of the control signal per drive line.
        ts: tf.float64
            Vector of times.
        gate: str
            Identifier for one of the gates.

        Returns
        -------
        unitary
            Matrix representation of the gate.
        """
        model = self.pmap.model
        h0, hctrls = model.get_Hamiltonians()
        signals = []
        hks = []
        for key in signal:
            signals.append(signal[key]["values"])
            hks.append(hctrls[key])
        dt = tf.constant(ts[1].numpy() - ts[0].numpy(), dtype=tf.complex128)

        if model.lindbladian:
            col_ops = model.get_Lindbladians()
            dUs = tf_utils.tf_propagation_lind(h0, hks, col_ops, signals, dt)
        else:
            dUs = tf_utils.tf_propagation(h0, hks, signals, dt)
        self.dUs[gate] = dUs
        self.ts = ts
        U = tf_utils.tf_matmul_left(dUs)
        self.U = U
        return U

    def set_opt_gates(self, seqs):
        """
        Specify a selection of gates to be computed.

        Parameters
        ----------
        opt_gates: Identifiers of the gates of interest. Can contain duplicates.

        """
        self.__opt_gates = list(set(itertools.chain.from_iterable(seqs)))

    def set_enable_dynamics_plots(self, flag, logdir):
        """
        Plotting of time-resolved populations.

        Parameters
        ----------
        flag: boolean
            Enable or disable plotting.
        logdir: str
            File path location for the resulting plots.
        """
        self.enable_dynamics_plots = flag
        self.logdir = logdir
        if self.enable_dynamics_plots:
            os.mkdir(self.logdir + "dynamics/")
            self.dynamics_plot_counter = 0

    def set_enable_pules_plots(self, flag, logdir):
        """
        Plotting of pulse shapes.

        Parameters
        ----------
        flag: boolean
            Enable or disable plotting.
        logdir: str
            File path location for the resulting plots.
        """
        self.enable_pulses_plots = flag
        self.logdir = logdir
        if self.enable_pulses_plots:
            os.mkdir(self.logdir + "pulses/")
            self.pulses_plot_counter = 0

    def set_enable_store_unitaries(self, flag, logdir):
        """
        Saving of unitary propagators.

        Parameters
        ----------
        flag: boolean
            Enable or disable saving.
        logdir: str
            File path location for the resulting unitaries.
        """
        self.enable_store_unitaries = flag
        self.logdir = logdir
        if self.enable_store_unitaries:
            os.mkdir(self.logdir + "unitaries/")
            self.store_unitaries_counter = 0

    def plot_dynamics(self, psi_init, seq, goal=-1, debug=False):
        # TODO double check if it works well
        """
        Plotting code for time-resolved populations.

        Parameters
        ----------
        psi_init: tf.Tensor
            Initial state or density matrix.
        seq: list
            List of operations to apply to the initial state.
        goal: tf.float64
            Value of the goal function, if used.
        debug: boolean
            If true, return a matplotlib figure instead of saving.
        """
        model = self.pmap.model
        dUs = self.dUs
        psi_t = psi_init.numpy()
        pop_t = self.populations(psi_t, model.lindbladian)
        for gate in seq:
            for du in dUs[gate]:
                psi_t = np.matmul(du.numpy(), psi_t)
                pops = self.populations(psi_t, model.lindbladian)
                pop_t = np.append(pop_t, pops, axis=1)
            if model.use_FR:
                instr = self.pmap.instructions[gate]
                signal, ts = self.pmap.generator.generate_signals(instr)
                # TODO change LO freq to at the level of a line
                freqs = {}
                framechanges = {}
                for line, ctrls in instr.comps.items():
                    offset = 0.0
                    if "gauss" in ctrls:
                        if ctrls['gauss'].params["amp"] != 0.0:
                            offset = ctrls['gauss'].params['freq_offset'].get_value()

                    freqs[line] = tf.cast(
                        ctrls['carrier'].params['freq'].get_value()
                        + offset,
                        tf.complex128
                    )
                    framechanges[line] = tf.cast(
                        ctrls['carrier'].params['framechange'].get_value(),
                        tf.complex128
                    )
                t_final = tf.constant(
                    instr.t_end - instr.t_start,
                    dtype=tf.complex128
                )
                FR = model.get_Frame_Rotation(
                    t_final,
                    freqs,
                    framechanges
                )
                if model.lindbladian:
                    FR = tf_utils.tf_super(FR)
                psi_t = tf.matmul(FR, psi_t)
                # TODO added framchanged psi to list

        fig, axs = plt.subplots(1, 1)
        ts = self.ts
        dt = ts[1] - ts[0]
        ts = np.linspace(0.0, dt*pop_t.shape[1], pop_t.shape[1])
        axs.plot(ts / 1e-9, pop_t.T)
        axs.grid(linestyle="--")
        axs.tick_params(
            direction="in", left=True, right=True, top=True, bottom=True
        )
        axs.set_xlabel('Time [ns]')
        axs.set_ylabel('Population')
        plt.legend(model.state_labels)
        if debug:
            plt.show()
        else:
            plt.savefig(self.logdir + f"dynamics/eval_{self.dynamics_plot_counter}_{seq[0]}_{goal}.png", dpi=300)

    def plot_pulses(self, instr, goal=-1, debug=False):
        """
        Plotting of pulse shapes.

        Parameters
        ----------
        instr : str
            Identifier of the current instruction.
        goal: tf.float64
            Value of the goal function, if used.
        debug: boolean
            If true, return a matplotlib figure instead of saving.
        """
        generator = self.pmap.generator
        signal, ts = generator.generate_signals(instr)
        awg = generator.devices["awg"]
        awg_ts = awg.ts

        if debug:
            pass
        else:
            # TODO Use os module to build paths
            foldername = self.logdir + "pulses/eval_" + str(self.pulses_plot_counter) + "_" + str(goal) + "/"
            if not os.path.exists(foldername):
                os.mkdir(foldername)
            os.mkdir(foldername + str(instr.name) + "/")

        fig, axs = plt.subplots(1, 1)

        for channel in instr.comps:
            inphase = awg.signal[channel]["inphase"]
            quadrature = awg.signal[channel]["quadrature"]
            axs.plot(awg_ts / 1e-9, inphase/1e-3, label="I " + channel)
            axs.plot(awg_ts / 1e-9, quadrature/1e-3, label="Q " + channel)
            axs.grid()
            axs.set_xlabel('Time [ns]')
            axs.set_ylabel('Pulse amplitude[mV]')
            plt.legend()
            if debug:
                pass
            else:
                with open(
                    self.logdir+f"pulses/eval_{self.pulses_plot_counter}_{goal}/{instr.name}/awg.log",
                    'a+'
                ) as logfile:
                    logfile.write(f"{channel}, inphase :\n")
                    logfile.write(json.dumps(inphase.numpy().tolist()))
                    logfile.write("\n")
                    logfile.write(f"{channel}, quadrature :\n")
                    logfile.write(json.dumps(quadrature.numpy().tolist()))
                    logfile.write("\n")
        if debug:
            plt.show()
        else:
            plt.savefig(
                self.logdir+f"pulses/eval_{self.pulses_plot_counter}_{goal}/{instr.name}/"
                f"awg_{list(instr.comps.keys())}.png",
                dpi=300
            )

        dac = generator.devices["dac"]
        dac_ts = dac.ts
        inphase = dac.signal["inphase"]
        quadrature = dac.signal["quadrature"]

        fig, axs = plt.subplots(1, 1)
        axs.plot(dac_ts / 1e-9, inphase/1e-3)
        axs.grid()
        axs.set_xlabel('Time [ns]')
        axs.set_ylabel('Pulse amplitude[mV]')
        if debug:
            plt.show()
        else:
            plt.savefig(
                self.logdir+f"pulses/eval_{self.pulses_plot_counter}_{goal}/{instr.name}/"
                f"dac_inphase_{list(instr.comps.keys())}.png", dpi=300
            )

        fig, axs = plt.subplots(1, 1)
        axs.plot(dac_ts / 1e-9, quadrature/1e-3)
        axs.grid()
        axs.set_xlabel('Time [ns]')
        axs.set_ylabel('Pulse amplitude[mV]')
        if debug:
            plt.show()
        else:
            plt.savefig(
                self.logdir+f"pulses/eval_{self.pulses_plot_counter}_{goal}/{instr.name}/"
                f"dac_quadrature_{list(instr.comps.keys())}.png", dpi=300
            )

        if "resp" in generator.devices:
            resp = generator.devices["resp"]
            resp_ts = dac_ts
            inphase = resp.signal["inphase"]
            quadrature = resp.signal["quadrature"]

            fig, axs = plt.subplots(1, 1)
            axs.plot(resp_ts / 1e-9, inphase/1e-3)
            axs.grid()
            axs.set_xlabel('Time [ns]')
            axs.set_ylabel('Pulse amplitude[mV]')
            if debug:
                plt.show()
            else:
                plt.savefig(
                    self.logdir+f"pulses/eval_{self.pulses_plot_counter}_{goal}/{instr.name}/"
                    f"resp_inphase_{list(instr.comps.keys())}.png", dpi=300
                )

            fig, axs = plt.subplots(1, 1)
            axs.plot(resp_ts / 1e-9, quadrature/1e-3)
            axs.grid()
            axs.set_xlabel('Time [ns]')
            axs.set_ylabel('Pulse amplitude[mV]')
            if debug:
                plt.show()
            else:
                plt.savefig(
                    self.logdir+f"pulses/eval_{self.pulses_plot_counter}_{goal}/{instr.name}/"
                    f"resp_quadrature_{list(instr.comps.keys())}.png", dpi=300
                )

        for channel in instr.comps:
            fig, axs = plt.subplots(1, 1)
            axs.plot(ts / 1e-9, signal[channel]["values"], label=channel)
            axs.grid()
            axs.set_xlabel('Time [ns]')
            axs.set_ylabel('signal')
            plt.legend()
        if debug:
            plt.show()
        else:
            plt.savefig(
                self.logdir+f"pulses/eval_{self.pulses_plot_counter}_{goal}/{instr.name}/"
                f"signal_{list(instr.comps.keys())}.png",
                dpi=300
            )

    def store_Udict(self, goal):
        """
        Save unitary as text and pickle.

        Parameter
        ---------
        goal: tf.float64
            Value of the goal function, if used.

        """
        folder = self.logdir + "unitaries/eval_" + str(self.store_unitaries_counter) + "_" + str(goal) + "/"
        if not os.path.exists(folder):
            os.mkdir(folder)
        with open(folder + 'Us.pickle', 'wb+') as file:
            pickle.dump(self.unitaries, file)
        for key, value in self.unitaries.items():
            np.savetxt(folder + key + ".txt", value)

    def populations(self, state, lindbladian):
        """
        Compute populations from a state or density vector.

        Parameters
        ----------
        state: tf.Tensor
            State or densitiy vector.
        lindbladian: boolean
            Specify if conversion to density matrix is needed.

        Returns
        -------
        tf.Tensor
            Vector of populations.
        """
        if lindbladian:
            rho = tf_utils.tf_vec_to_dm(state)
            pops = tf.math.real(tf.linalg.diag_part(rho))
            return tf.reshape(pops, shape=[pops.shape[0], 1])
        else:
            return tf.abs(state)**2
