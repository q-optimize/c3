"""Simulator."""

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

from c3po.experiment import Experiment
from c3po.control import GateSet

from c3po.tf_utils import tf_propagation
from c3po.tf_utils import tf_propagation_lind
from c3po.tf_utils import tf_matmul_list
from c3po.tf_utils import tf_super
from c3po.qt_utils import single_length_RB


class Simulator():
    """Simulator object."""

    def __init__(self,
                 exp: Experiment,
                 gateset: GateSet
                 ):
        self.exp = exp
        self.gateset = gateset
        self.unitaries = {}
        self.lindbladian = False

    def get_gates(
        self,
        gateset_values: list,
        gateset_opt_map: list
    ):
        gates = {}
        self.gateset.set_parameters(gateset_values, gateset_opt_map)
        # TODO allow for not passing model params
        # model_params, _ = self.model.get_values_bounds()
        for gate in self.gateset.instructions.keys():
            signal = self.exp.generator.generate_signals(
                self.gateset.instructions[gate]
            )
            U = self.propagation(signal)
            gates[gate] = U
            self.unitaries = gates
        return gates

    def evaluate_sequences(
        self,
        U_dict: dict,
        sequences: list
    ):
        gates = U_dict
        # TODO deal with the case where you only evaluate one sequence
        U = []
        for sequence in sequences:
            Us = []
            for gate in sequence:
                Us.append(gates[gate])
            U.append(tf_matmul_list(Us))
        return U

    def propagation(self,
                    signal: dict
                    ):
        signals = []
        # This sorting ensures that signals and hks are matched
        # TODO do hks and signals matching more rigorously
        for key in sorted(signal):
            out = signal[key]
            # TODO this points to the fact that all sim_res must be the same
            ts = out["ts"]
            signals.append(out["values"])

<<<<<<< HEAD
        dt = tf.cast(ts[1] - ts[0], tf.complex128, name="dt")
        h0, hks = self.exp.get_Hamiltonians()
        if lindbladian:
            col_ops = self.exp.get_lindbladian()
=======
        dt = ts[1].numpy() - ts[0].numpy()
        h0, hks = self.exp.model.get_Hamiltonians()
        if self.lindbladian:
            col_ops = self.exp.model.get_lindbladian()
>>>>>>> 8435f4bf720b59c7a8b100cd69946b10ad946eef
            dUs = tf_propagation_lind(h0, hks, col_ops, signals, dt)
        else:
            dUs = tf_propagation(h0, hks, signals, dt)
        self.dUs = dUs
        self.ts = ts
        U = tf_matmul_list(dUs)
        if hasattr(self, 'VZ'):
            if self.lindbladian:
                U = tf.matmul(tf_super(self.VZ), U)
            else:
                U = tf.matmul(self.VZ, U)
        self.U = U
        return U

    def plot_dynamics(self,
                      psi_init
                      ):
        # TODO double check if it works well
        dUs = self.dUs
        psi_t = psi_init.numpy()
        pop_t = self.populations(psi_t)
        for du in dUs:
            psi_t = np.matmul(du.numpy(), psi_t)
            pops = self.populations(psi_t)
            pop_t = np.append(pop_t, pops, axis=1)
        fig, axs = plt.subplots(1, 1)
        ts = self.ts
        dt = ts[1] - ts[0]
        ts = np.append(0, ts + dt / 2)
        axs.plot(ts / 1e-9, pop_t.T)
        axs.grid()
        axs.set_xlabel('Time [ns]')
        axs.set_ylabel('Population')
        fig.show()

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

        # PLOT
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
        fitted = RB_fit(lengths, r, A, B)
        ax.plot(lengths, fitted)
        plt.text(0.1, 0.1,
                 'r={:.4f}, A={:.3f}, B={:.3f}'.format(r, A, B),
                 size=16,
                 transform=ax.transAxes)
        plt.title('RB results')
        plt.ylabel('Population in 0')
        plt.xlabel('# Cliffords')
        plt.ylim(0, 1)
        plt.xlim(0, lengths[-1])
        plt.show()
        return r, A, B
