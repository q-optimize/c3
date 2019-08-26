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

    def __init__(self, model, generator):
        self.model = model
        self.generator = generator


    def propagation():
        signals = self.generator.generate_signals()
        #dt = tf.cast(ts[1], tf.complex128)
        h0 = tf.cast(sum(self.model.drift_Hs), tf.complex128)
        hks = [ tf.cast(control_H, tf.complex128) for control_H
        hamiltonians = self.model.get_hamiltonians(signals)
        utils.tf_propagation(
            h0,
            signals,
                             )
        return ...
