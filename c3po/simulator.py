"""Simulator."""

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

from c3po.experiment import Experiment
from c3po.control import GateSet

from c3po.tf_utils import tf_propagation as tf_propagation
from c3po.tf_utils import tf_propagation_lind as tf_propagation_lind
from c3po.tf_utils import tf_matmul_list as tf_matmul_list
from c3po.tf_utils import tf_super as tf_super


class Simulator():
    """Simulator object."""

    def __init__(self,
                 exp: Experiment,
                 gateset: GateSet
                 # controls: GateSet
                 ):
        self.exp = exp
        self.gateset = gateset
        self.unitaries = {}

    def get_gates(
        self,
        gateset_values: list,
        gateset_opt_map: list,
        lindbladian: bool = False
    ):
        gates = {}
        self.gateset.set_parameters(gateset_values, gateset_opt_map)
        # TODO allow for not passing model params
        # model_params, _ = self.model.get_values_bounds()
        for gate in self.gateset.instructions.keys():
            signal = self.exp.generator.generate_signals(
                self.gateset.instructions[gate]
            )
            U = self.propagation(signal, lindbladian)
            gates[gate] = U
            if hasattr(self, 'VZ'):
                if lindbladian:
                    gates[gate] = tf.matmul(tf_super(self.VZ), gates[gate])
                else:
                    gates[gate] = tf.matmul(self.VZ, gates[gate])
            self.unitaries = gates
        return gates

    def evaluate_sequence(
        self,
        gateset_values: list,
        gateset_opt_map: list,
        sequence: list,
        lindbladian: bool = False
    ):
        gates = self.get_gates(
            gateset_values,
            gateset_opt_map,
            lindbladian
        )
        Us = []
        for gate in sequence:
            Us.append(gates[gate])
        U = tf_matmul_list(Us)
        return U

    def propagation(self,
                    signal: dict,
                    lindbladian: bool = False
                    ):
        signals = []
        # This sorting ensures that signals and hks are matched
        # TODO do hks and signals matching more rigorously
        for key in sorted(signal):
            out = signal[key]
            # TODO this points to the fact that all sim_res must be the same
            ts = out["ts"]
            signals.append(out["values"])

        dt = tf.cast(ts[1] - ts[0], tf.complex128, name="dt")
        h0, hks = self.exp.model.get_Hamiltonians()
        if lindbladian:
            col_ops = self.exp.model.get_lindbladian()
            dUs = tf_propagation_lind(h0, hks, col_ops, signals, dt)
        else:
            dUs = tf_propagation(h0, hks, signals, dt)
        self.dUs = dUs
        self.ts = ts
        U = tf_matmul_list(dUs)
        self.U = U
        return U

    def plot_dynamics(self,
                      psi_init,
                      lindbladian: bool = False
                      ):
        # TODO double check if it works well
        dUs = self.dUs
        psi_t = psi_init.numpy()
        pop_t = self.populations(psi_t, dv=lindbladian)
        for du in dUs:
            psi_t = np.matmul(du.numpy(), psi_t)
            pops = self.populations(psi_t, dv=lindbladian)
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

    @staticmethod
    def populations(state, dv=False):
        if dv:
            dim = int(np.sqrt(len(state)))
            indeces = [n * dim + n for n in range(dim)]
            return np.abs(state[indeces])
        else:
            return np.abs(state)**2
