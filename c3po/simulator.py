"""Simulator."""

import os
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

from c3po.experiment import Experiment
from c3po.control import GateSet

from c3po.tf_utils import tf_propagation
from c3po.tf_utils import tf_propagation_lind
from c3po.tf_utils import tf_matmul_left, tf_matmul_right
from c3po.tf_utils import tf_super


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

    def get_gates(self):
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
                FR = self.exp.model.get_Frame_Rotation(
                    t_final,
                    lo_freqs,
                    framechanges
                )
                if self.lindbladian:
                    SFR = tf_super(FR)
                    U = tf.matmul(SFR, U)
                    self.FR = SFR
                else:
                    U = tf.matmul(FR, U)
                    self.FR = FR
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
            U.append(tf_matmul_left(Us))

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

    def plot_dynamics(self, psi_init, seq, data_path="/tmp/c3figs/"):
        # TODO double check if it works well
        dUs = self.dUs
        psi_t = psi_init.numpy()
        pop_t = self.populations(psi_t)
        for gate in seq:
            for du in dUs[gate]:
                psi_t = np.matmul(du.numpy(), psi_t)
                pops = self.populations(psi_t)
                pop_t = np.append(pop_t, pops, axis=1)
            if self.use_VZ:
                psi_t = tf.matmul(self.FR, psi_t)

        fig, axs = plt.subplots(1, 1)
        ts = self.ts
        dt = ts[1] - ts[0]
        ts = np.linspace(0.0, dt*pop_t.shape[1], pop_t.shape[1])
        axs.plot(ts / 1e-9, pop_t.T)
        axs.grid()
        axs.set_xlabel('Time [ns]')
        axs.set_ylabel('Population')
        plt.legend(self.exp.model.state_labels, loc="center left")
        if not os.path.isdir(data_path):
            os.makedirs(data_path)
        fig.savefig(data_path+'dynamics.png', dpi=300)
        #plt.show(block=False)
        plt.close(fig)

    def populations(self, state):
        if self.lindbladian:
            diag = []
            dim = int(np.sqrt(len(state)))
            indeces = [n * dim + n for n in range(dim)]
            for indx in indeces:
                diag.append(state[indx])
            return np.abs(diag)
        else:
            return np.abs(state)**2
