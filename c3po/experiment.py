"""
Experiment class that models and simulates the whole experiment.

It combines the information about the model of the quantum device, the control stack and the operations that can be
done on the device.

Given this information an experiment run is simulated, returning either processes, states or populations.
"""

import os
import json
import pickle
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import c3po.utils.tf_utils as tf_utils


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

    def __init__(self, model=None, generator=None, gateset=None):
        self.generator = generator
        self.gateset = gateset

        self.unitaries = {}
        self.dUs = {}

        components = {}
        if model:
            self.model = model
            components.update(self.model.couplings)
            components.update(self.model.subsystems)
            components.update(self.model.tasks)
        components.update(self.generator.devices)
        self.components = components

        id_list = []
        par_lens = []
        for comp in self.components.values():
            id_list.extend(comp.list_parameters())
            for par in comp.params.values():
                par_lens.append(par.length)
        self.id_list = id_list
        self.par_lens = par_lens

    def write_config(self):
        """
        Return the current experiment as a JSON compatible dict.

        EXPERIMENTAL
        """
        cfg = {}
        cfg['model'] = self.model.write_config()
        cfg['generator'] = self.generator.write_config()
        cfg['gateset'] = self.gateset.write_config()
        return cfg

    def get_parameters(self, opt_map=None, scaled=False):
        """
        Return the current parameters.

        Parameters
        ----------
        opt_map: tuple
            Hierarchical identifier for parameters.
        scaled: boolean
            If true, return the optimizer friendly version. See Quantity.

        """
        if opt_map is None:
            opt_map = self.id_list
        values = []
        for id in opt_map:
            comp_id = id[0]
            par_id = id[1]
            par = self.components[comp_id].params[par_id]
            if scaled:
                values.extend(par.get_opt_value())
            else:
                values.append(par.get_value())
        return values

    def set_parameters(self, values: list, opt_map: list, scaled=False):
        """Set the values in the original instruction class.

        Parameters
        ----------
        values: list
            List of parameter values.
        opt_map: list
            Corresponding identifiers for the parameter values.

        """
        val_indx = 0
        for id in opt_map:
            comp_id = id[0]
            par_id = id[1]
            id_indx = self.id_list.index(id)
            par_len = self.par_lens[id_indx]
            par = self.components[comp_id].params[par_id]
            if scaled:
                par.set_opt_value(values[val_indx:val_indx+par_len])
                val_indx += par_len
            else:
                try:
                    par.set_value(values[val_indx])
                    val_indx += 1
                except ValueError:
                    raise ValueError(f"Trying to set {id} to value {values[val_indx]}")
        self.model.update_model()

    def print_parameters(self, opt_map=None):
        """
        Return a multi-line human-readable string of the parameter names and
        current values.

        Parameters
        ----------
        opt_map: list
            Optionally use only the specified parameters.

        """
        ret = []
        if opt_map is None:
            opt_map = self.id_list
        for id in opt_map:
            comp_id = id[0]
            par_id = id[1]
            par = self.components[comp_id].params[par_id]
            nice_id = f"{comp_id}-{par_id}"
            ret.append(f"{nice_id:32}: {par}\n")
        return "".join(ret)

    def evaluate(self, seqs):
        """
        Compute the population values for a given sequence of operations.

        Parameters
        ----------
        seqs: str list
            A list of control pulses/gates to perform on the device.

        """
        Us = tf_utils.evaluate_sequences(self.unitaries, seqs)
        psi_init = self.model.tasks["init_ground"].initialise(
            self.model.drift_H,
            self.model.lindbladian
        )
        self.psi_init = psi_init
        populations = []
        for U in Us:
            psi_final = tf.matmul(U, self.psi_init)
            pops = self.populations(
                psi_final, self.model.lindbladian
            )
            populations.append(pops)
        self.pops = populations

    def process(self, labels=None):
        """
        Apply a readout procedure to a population vector. Very specialized
        at the moment.

        Parameters
        ----------
        labels: list
            List of state labels specifying a subspace.

        Returns
        -------
        list
            A list of processed populations.

        """
        populations_final = []
        for pops in self.pops:
            # TODO: Loop over all tasks in a general fashion
            if "conf_matrix" in self.model.tasks:
                pops = self.model.tasks["conf_matrix"].confuse(pops)
            if labels is not None:
                pops_select = 0
                for label in labels:
                    pops_select += pops[
                        self.model.comp_state_labels.index(label)
                    ]
                pops = pops_select
            else:
                pops = tf.reshape(pops, [pops.shape[0]])

            if "meas_rescale" in self.model.tasks:
                pops = self.model.tasks["meas_rescale"].rescale(pops)
            populations_final.append(pops)
        return populations_final

    def get_gates(self):
        """
        Compute the unitary representation of operations. If no operations are
        specified in self.opt_gates the complete gateset is computed.

        Returns
        -------
        dict
            A dictionary of gate names and their unitary representation.
        """
        gates = {}
        gate_keys = self.gateset.instructions.keys()
        if "opt_gates" in self.__dict__:
            if self.opt_gates:
                gate_keys = self.opt_gates
        for gate in gate_keys:
#             print(gate)
            instr = self.gateset.instructions[gate]
            signal, ts = self.generator.generate_signals(instr)
            U = self.propagation(signal, ts, gate)
            if self.model.use_FR:
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
                FR = self.model.get_Frame_Rotation(
                    t_final,
                    freqs,
                    framechanges
                )
                print(repr(FR))
                if self.model.lindbladian:
                    SFR = tf_utils.tf_super(FR)
                    U = tf.matmul(SFR, U)
                    self.FR = SFR
                else:
                    U = tf.matmul(FR, U)
                    self.FR = FR
            if self.model.dephasing_strength != 0.0:
                if not self.model.lindbladian:
                    raise ValueError(
                        'Dephasing can only be added when lindblad is on.'
                    )
                else:
                    amps = {}
                    for line, ctrls in instr.comps.items():
                        amp, sum = self.generator.devices['awg'].get_average_amp(line)
                        amps[line] = tf.cast(amp, tf.complex128)
                    t_final = tf.constant(
                        instr.t_end - instr.t_start,
                        dtype=tf.complex128
                    )
                    dephasing_channel = self.model.get_dephasing_channel(
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
        h0, hctrls = self.model.get_Hamiltonians()
        signals = []
        hks = []
        for key in signal:
            signals.append(signal[key]["values"])
            hks.append(hctrls[key])
        dt = ts[1].numpy() - ts[0].numpy()

        if self.model.lindbladian:
            col_ops = self.model.get_Lindbladians()
            dUs = tf_utils.tf_propagation_lind(h0, hks, col_ops, signals, dt)
        else:
            dUs = tf_utils.tf_propagation(h0, hks, signals, dt)
        self.dUs[gate] = dUs
        self.ts = ts
        U = tf_utils.tf_matmul_left(dUs)
        self.U = U
        return U

    def set_opt_gates(self, opt_gates):
        """
        Specify a selection of gates to be computed.

        Parameters
        ----------
        opt_gates: Identifiers of the gates of interest.

        """
        self.opt_gates = opt_gates

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

    def plot_dynamics(self, psi_init, seq, goal=None, debug=False, oper=None):
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
        dUs = self.dUs
        psi_t = psi_init.numpy()
        pop_t = self.populations(psi_t, self.model.lindbladian, oper=oper)
        dt = self.ts[1] - self.ts[0]
        times = np.array([0.0])
        for gate in seq:
            for du in dUs[gate]:
                psi_t = np.matmul(du.numpy(), psi_t)
                pops = self.populations(psi_t, self.model.lindbladian, oper=oper)
                pop_t = np.append(pop_t, pops, axis=1)
                times = np.append(times, times[-1]+dt)
            if self.model.use_FR:
                instr = self.gateset.instructions[gate]
                signal, ts = self.generator.generate_signals(instr)
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
                FR = self.model.get_Frame_Rotation(
                    t_final,
                    freqs,
                    framechanges
                )
                if self.model.lindbladian:
                    FR = tf_utils.tf_super(FR)
                psi_t = tf.matmul(FR, psi_t)
                pops = self.populations(psi_t, self.model.lindbladian, oper=oper)
                pop_t = np.append(pop_t, pops, axis=1)
                times = np.append(times, times[-1])

        fig, axs = plt.subplots(1, 1)
        axs.plot(times / 1e-9, pop_t.T, '-')
        axs.grid(linestyle="--")
        axs.tick_params(
            direction="in", left=True, right=True, top=True, bottom=True
        )
        axs.set_xlabel('Time [ns]')
        axs.set_ylabel('Population')
        plt.legend(self.model.state_labels)
        if debug:
            plt.show()
        else:
            plt.savefig(
                self.logdir +
                f"dynamics/eval_{self.dynamics_plot_counter}_{seq[0]}_{goal}.png",
                dpi=300
            )
        plt.close("all")

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
        signal, ts = self.generator.generate_signals(instr)
        awg = self.generator.devices["awg"]
        awg_ts = awg.ts

#         print(instr.name)
#         print(instr.t_end)
#         print(awg_ts)

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

        dac = self.generator.devices["dac"]
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

        if "resp" in self.generator.devices:
            resp = self.generator.devices["resp"]
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
        plt.close("all")

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

    def populations(self, state, lindbladian, oper=None):
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
        if oper is not None:
            if lindbladian:
                rho = tf_utils.tf_vec_to_dm(state)
            else:
                rho = tf_utils.tf_state_to_dm(state)
            trace = np.trace(np.matmul(rho,oper))
            return [[np.real(trace)]] #,[np.imag(trace)]]
        else:
            if lindbladian:
                rho = tf_utils.tf_vec_to_dm(state)
                pops = tf.math.real(tf.linalg.diag_part(rho))
                return tf.reshape(pops, shape=[pops.shape[0], 1])
            else:
                return tf.abs(state)**2
