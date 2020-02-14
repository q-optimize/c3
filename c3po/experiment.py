"""Experiment class that models the whole experiment."""

import copy
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import c3po.tf_utils as tf_utils


class Experiment:
    """
    It models all of the behaviour of the physical experiment.

    It contains boxes that perform a part of the experiment routine.

    Parameters
    ----------
    model: Model
    generator: Generator

    """

    def __init__(self, model, generator, gateset):
        self.model = model
        self.generator = generator
        self.gateset = gateset

        self.unitaries = {}
        self.dUs = {}

        components = {}
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
        cfg = {}
        cfg['model'] = self.model.write_config()
        cfg['generator'] = self.generator.write_config()
        cfg['gateset'] = self.gateset.write_config()
        return cfg

    def get_parameters(self, opt_map=None, scaled=False):
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
        """Set the values in the original instruction class."""
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
                    print("Value out of bounds")
                    print(f"Trying to set {id} to value {values[val_indx]}")
        self.model.update_model()

    def print_parameters(self, opt_map=None):
        if opt_map is None:
            opt_map = self.list_parameters()
        for id in opt_map:
            comp_id = id[0]
            par_id = id[1]
            self.components[comp_id].print_parameter(par_id)

    # THE ROLE OF THE OLD SIMULATOR AND OTHERS

    def evaluate(self, seqs):
        U_dict = self.get_gates()
        psi_init = self.model.tasks["init_ground"].initialise(
            self.model.drift_H,
            self.model.lindbladian
        )
        self.psi_init = psi_init
        Us = self.evaluate_sequences(U_dict, seqs)
        pop1s = []
        for U in Us:
            psi_final = tf.matmul(U, psi_init)
            pops = self.populations(psi_final, self.model.lindbladian)
            pop1 = self.model.tasks["conf_matrix"].pop1(
                pops,
                self.model.lindbladian
            )
            pop1 = self.model.tasks["meas_rescale"].rescale(pop1)
            pop1s.append(pop1)
        return pop1s

    def get_gates(self):
        gates = {}
        # TODO allow for not passing model params
        # model_params, _ = self.model.get_values_bounds()
        for gate in self.gateset.instructions.keys():
            instr = self.gateset.instructions[gate]
            signal, ts = self.generator.generate_signals(instr)
            U = self.propagation(signal, ts, gate)
            if self.model.use_FR:
                # TODO change LO freq to at the level of a line
                lo_freqs = {}
                framechanges = {}
                for line, ctrls in instr.comps.items():
                    lo_freqs[line] = tf.cast(
                        ctrls['carrier'].params['freq'].get_value(),
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
                    lo_freqs,
                    framechanges
                )
                if self.model.lindbladian:
                    SFR = tf_utils.tf_super(FR)
                    U = tf.matmul(SFR, U)
                    self.FR = SFR
                else:
                    U = tf.matmul(FR, U)
                    self.FR = FR
            gates[gate] = U
            self.unitaries = gates
        return gates

    @staticmethod
    def evaluate_sequences(
        U_dict: dict,
        sequences: list
    ):
        """
        Sequences are assumed to be given in the correct order (left to right).

            e.g.
            ['X90p','Y90p','Xp'] --> U = X90p x Y90p x Xp
        """
        gates = U_dict
        # TODO deal with the case where you only evaluate one sequence
        U = []
        for sequence in sequences:
            Us = []
            for gate in sequence:
                Us.append(gates[gate])
            U.append(tf_utils.tf_matmul_left(Us))
        return U

    def propagation(
        self,
        signal: dict,
        ts,
        gate
    ):

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

    def plot_dynamics(self, psi_init, seq):
        # TODO double check if it works well
        dUs = self.dUs
        psi_t = psi_init.numpy()
        pop_t = self.populations(psi_t, self.model.lindbladian)
        for gate in seq:
            for du in dUs[gate]:
                psi_t = np.matmul(du.numpy(), psi_t)
                pops = self.populations(psi_t, self.model.lindbladian)
                pop_t = np.append(pop_t, pops, axis=1)
            if self.model.use_FR:
                psi_t = tf.matmul(self.FR, psi_t)

        fig, axs = plt.subplots(1, 1)
        ts = self.ts
        dt = ts[1] - ts[0]
        ts = np.linspace(0.0, dt*pop_t.shape[1], pop_t.shape[1])
        axs.plot(ts / 1e-9, pop_t.T)
        axs.grid()
        axs.set_xlabel('Time [ns]')
        axs.set_ylabel('Population')
        return fig, axs

    def populations(self, state, lindbladian):
        if lindbladian:
            rho = tf_utils.tf_vec_to_dm(state)
            return tf.math.real(tf.linalg.diag_part(rho))
        else:
            return tf.abs(state)**2
