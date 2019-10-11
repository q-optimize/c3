import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from c3po.utils.tf_utils import tf_propagation as tf_propagation
from c3po.utils.tf_utils import tf_propagation_lind as tf_propagation_lind
from c3po.utils.tf_utils import tf_matmul_list as tf_matmul_list

class Simulator():
    """Short summary.

    Parameters
    ----------
    model : type
        Description of parameter `model`.
    solve_func : type
        Description of parameter `solve_func`.

    Attributes
    ----------
    solver : func
        Function that integrates the Schrödinger/Lindblad/Whatever EOM
    resolution : float
        Determined by numerical accuracy. We'll compute this from something
        like highest eigenfrequency, etc.
    model : Model
        Class that symbolically describes the model.
    """

    def __init__(self, model, generator, controls):
        self.model = model
        self.generator = generator
        self.controls = controls
        plt.rcParams['figure.dpi'] = 100
        fig, axs = plt.subplots(1, 1)
        plt.ion()
        plt.show()
        self.fig = fig
        self.axs = axs

    def propagation(self,
                    pulse_params,
                    opt_params,
                    model_params = None,
                    lindbladian = False
                    ):
        self.controls.update_controls(pulse_params, opt_params)
        gen_output = self.generator.generate_signals(self.controls.controls)
        self.generator.devices['awg'].plot_IQ_components(self.fig, self.axs)
        signals = []

        for key in gen_output:
            out = gen_output[key]
            ts = out["ts"]
            signals.append(out["signal"])

        dt = tf.cast(ts[1]-ts[0], tf.complex128, name="dt")
        if model_params is not None:
            h0, hks = self.model.get_Hamiltonians(model_params)
        else:
            h0, hks = self.model.get_Hamiltonians()
        if lindbladian:
            col_op = self.model.get_lindbladian()
            dUs =  tf_propagation_lind(h0, hks, col_op, signals, dt)
        else:
            dUs =  tf_propagation(h0, hks, signals, dt)
        self.dUs = dUs
        self.ts = ts
        U = tf_matmul_list(dUs)
        return U

    def plot_dynamics(self, psi_init):
        dUs = self.dUs
        psi_t = psi_init.numpy()
        pop_t = np.abs(psi_t)**2
        for du in dUs:
            psi_t = np.matmul(du.numpy(),psi_t)
            pop_t = np.append(pop_t, np.abs(psi_t)**2 ,axis=1)
        fig, axs = plt.subplots(1, 1)
        plt.ion()
        plt.plot(pop_t.T)
        plt.grid()
        plt.show(block=False)
