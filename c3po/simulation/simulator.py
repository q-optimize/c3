import tensorflow as tf
from c3po.utils.tf_utils import tf_propagation as tf_propagation

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


    def propagation(self, params, opt_params):
        self.controls.update_controls(params, opt_params)
        gen_output = self.generator.generate_signals(self.controls.controls)
        signals = []
        for key in gen_output:
            out = gen_output[key]
            ts = out["ts"]
            signals.append(out["signal"])
        dt = tf.cast(ts[1]-ts[0], tf.complex128, name="dt")
        h0, hks = self.model.get_Hamiltonians()
        U = tf_propagation(h0, hks, signals, dt)
        return U
