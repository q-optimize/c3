"""Simulator."""

import os
import copy
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

from c3po.experiment import Experiment
from c3po.control import GateSet

from c3po.tf_utils import tf_propagation
from c3po.tf_utils import tf_propagation_lind
from c3po.tf_utils import tf_matmul_left, tf_matmul_right
from c3po.tf_utils import tf_super
from c3po.qt_utils import single_length_RB


# TODO resctucture so that sim,mdl,gen,meas,gateset are all in one class (exp?)
class Simulator():
    """Simulator object."""

    def __init__(
        self,
        exp: Experiment,
        gateset: GateSet
    ):
        self.exp = exp
        self.gateset = gateset
        self.unitaries = {}
        self.lindbladian = False
        self.dUs = {}

    def write_config(self):
        return 0

    def get_gates(
        self
    ):
        gates = {}
        # TODO allow for not passing model params
        # model_params, _ = self.model.get_values_bounds()
        for gate in self.gateset.instructions.keys():
            instr = self.gateset.instructions[gate]
            signal, ts = self.exp.generator.generate_signals(instr)
            U = self.propagation(signal, ts, gate)
            if self.use_VZ:
                # TODO change LO freq to at the level of a line
                lo_freqs = {}
                for line, ctrls in instr.comps.items():
                    lo_freqs[line] = tf.cast(
                        ctrls['carrier'].params['freq'].get_value(),
                        tf.complex128
                    )
                t_final = tf.constant(
                    instr.t_end - instr.t_start,
                    dtype=tf.complex128
                )
                FR = self.exp.model.get_Frame_Rotation(t_final, lo_freqs)
                if self.lindbladian:
                    U = tf.matmul(tf_super(FR), U)
                else:
                    U = tf.matmul(FR, U)
            gates[gate] = U
            self.unitaries = gates
        return gates

    def evaluate_sequences(
        self,
        U_dict: dict,
        sequences: list
    ):
        """
        Sequences are assumed to be given in the correct order (left to right).
            e.g.
            ['X90p','Y90p'] --> U = X90p x Y90p
        """
        gates = U_dict
        # TODO deal with the case where you only evaluate one sequence
        U = []
        for sequence in sequences:
            Us = []
            for gate in sequence:
                Us.append(gates[gate])
            U.append(tf_matmul_right(Us))
        return U

    def propagation(
        self,
        signal: dict,
        ts,
        gate
    ):

        h0, hctrls = self.exp.model.get_Hamiltonians()
        signals = []
        hks = []
        for key in signal:
            signals.append(signal[key]["values"])
            hks.append(hctrls[key])
        dt = ts[1].numpy() - ts[0].numpy()

        if self.lindbladian:
            col_ops = self.exp.model.get_Lindbladians()
            dUs = tf_propagation_lind(h0, hks, col_ops, signals, dt)
        else:
            dUs = tf_propagation(h0, hks, signals, dt)
        self.dUs[gate] = dUs
        self.ts = ts
        U = tf_matmul_left(dUs)
        self.U = U
        return U

    def plot_dynamics(self, psi_init, seq):
        # TODO double check if it works well
        dUs = self.dUs
        psi_t = psi_init.numpy()
        pop_t = self.populations(psi_t)
        for gate in seq:
            for du in dUs[gate]:
                psi_t = np.matmul(du.numpy(), psi_t)
                pops = self.populations(psi_t)
                pop_t = np.append(pop_t, pops, axis=1)
        fig, axs = plt.subplots(1, 1)
        ts = self.ts
        dt = ts[1] - ts[0]
        ts = np.linspace(0.0, dt*pop_t.shape[1], pop_t.shape[1])
        axs.plot(ts / 1e-9, pop_t.T)
        axs.grid()
        axs.set_xlabel('Time [ns]')
        axs.set_ylabel('Population')
        data_path = "/localdisk/c3logs/recent/"
        if not os.path.isdir(data_path):
            os.makedirs(data_path)
        fig.savefig(data_path+'dynamics.png')
        plt.close()

    def populations(self, state):
        if self.lindbladian:
            dim = int(np.sqrt(len(state)))
            indeces = [n * dim + n for n in range(dim)]
            return np.abs(state[indeces])
        else:
            return np.abs(state)**2

    def RB(self,
           psi_init,
           U_dict,
           min_length: int = 5,
           max_length: int = 100,
           num_lengths: int = 20,
           num_seqs: int = 30,
           plot_all=True,
           progress=True,
           logspace=False
           ):
        print('performing RB experiment')
        if logspace:
            lengths = np.rint(
                        np.logspace(
                            np.log10(min_length),
                            np.log10(max_length),
                            num=num_lengths
                            )
                        ).astype(int)
        else:
            lengths = np.rint(
                        np.linspace(
                            min_length,
                            max_length,
                            num=num_lengths
                            )
                        ).astype(int)
        surv_prob = []
        for L in lengths:
            if progress:
                print(L)
            seqs = single_length_RB(num_seqs, L)
            Us = self.evaluate_sequences(U_dict, seqs)
            pop0s = []
            for U in Us:
                pops = self.populations(tf.matmul(U, psi_init))
                pop0s.append(float(pops[0]))
            surv_prob.append(pop0s)

        def RB_fit(len, r, A, B):
            return A * r**(len) + B
        bounds = (0, 1)
        init_guess = [0.9, 0.5, 0.5]
        fitted = False
        while not fitted:
            print('lengths shape', lengths.shape)
            print('surv_prob shape', np.array(surv_prob).shape)
            try:
                means = np.mean(surv_prob, axis=1)
                stds = np.std(surv_prob, axis=1) / np.sqrt(len(surv_prob[0]))
                solution, cov = curve_fit(RB_fit,
                                          lengths,
                                          means,
                                          sigma=stds,
                                          bounds=bounds,
                                          p0=init_guess)
                r, A, B = solution
                fitted = True
            except Exception as message:
                print(message)
                print('increasing RB length.')
                if logspace:
                    new_lengths = np.rint(
                                np.logspace(
                                    np.log10(max_length + min_length),
                                    np.log10(max_length * 2),
                                    num=num_lengths
                                    )
                                ).astype(int)
                else:
                    new_lengths = np.rint(
                                np.linspace(
                                    max_length + min_length,
                                    max_length*2,
                                    num=num_lengths
                                    )
                                ).astype(int)
                max_length = max_length * 2
                for L in new_lengths:
                    if progress:
                        print(L)
                    seqs = single_length_RB(num_seqs, L)
                    Us = self.evaluate_sequences(U_dict, seqs)
                    pop0s = []
                    for U in Us:
                        pops = self.populations(tf.matmul(U, psi_init))
                        pop0s.append(float(pops[0]))
                    surv_prob.append(pop0s)
                lengths = np.append(lengths, new_lengths)

            # PLOT inside the while loop
            fig, ax = plt.subplots()
            if plot_all:
                ax.plot(lengths,
                        surv_prob,
                        marker='o',
                        color='red',
                        linestyle='None')
            ax.errorbar(lengths,
                        means,
                        yerr=stds,
                        color='blue',
                        marker='x',
                        linestyle='None')
            plt.title('RB results')
            plt.ylabel('Population in 0')
            plt.xlabel('# Cliffords')
            plt.ylim(0, 1)
            plt.xlim(0, lengths[-1])
            plt.show()
        # put the fitted line only once fit parameters have been found
        fitted = RB_fit(lengths, r, A, B)
        ax.plot(lengths, fitted)
        plt.text(0.1, 0.1,
                 'r={:.4f}, A={:.3f}, B={:.3f}'.format(r, A, B),
                 size=16,
                 transform=ax.transAxes)
        return r, A, B
