import tensorflow as tf
from c3po.utils.tf_utils import tf_propagation as tf_propagation
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

    def propagation(self, pulse_params, opt_params, model_params = None):

        self.controls.update_controls(pulse_params, opt_params)
        gen_output = self.generator.generate_signals(self.controls.controls)
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
        dUs =  tf_propagation(h0, hks, signals, dt)
        U = tf_matmul_list(dUs)
        return U
