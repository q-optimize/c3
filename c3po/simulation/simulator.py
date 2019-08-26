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

    def __init__(self, model):
        self.model = model

    def propagation(self, U0, gate, params, history=False):
        cflds, ts = gate.get_control_fields(params, self.resolution)
        h0 = self.model.tf_H0
        hks = self.model.tf_Hcs
        dt = tf.cast(ts[1], tf.complex128)
        def dU_of_t(cflds_t):
            h = h0
            for ii in range(len(hks)):
                    h += cflds_t[ii]*hks[ii]
            return tf.linalg.expm(-1j*h*dt)

        cf = tf.cast(
            tf.transpose(tf.stack(cflds)),
            tf.complex128,
            name='Control_fields'
            )

        dUs = tf.map_fn(
            dU_of_t,cf,
            name='dU_of_t'
            )

        if history:
            u_t = tf.gather(dUs,0)
            history = [u_t]
            for ii in range(dUs.shape[0]-1):
                du = tf.gather(dUs, ii+1)
                u_t = tf.matmul(du,u_t)
                history.append(u_t)
            return history, ts
        else:
            U = tf.gather(dUs, 0)
            for ii in range(1, dUs.shape[0]):
                U = tf.matmul(tf.gather(dUs, ii), U)
            return U
