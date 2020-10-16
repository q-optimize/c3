from c3.c3objs import C3obj, Quantity
import tensorflow as tf
import numpy as np
import c3.libraries.constants as constants
import c3.utils.tf_utils as tf_utils
import c3.utils.qt_utils as qt_utils


class Task(C3obj):
    # TODO BETTER NAME FOR TASK!
    """Task that is part of the measurement setup."""

    def __init__(
            self,
            name: str = " ",
            desc: str = " ",
            comment: str = " ",
    ):
        super().__init__(
            name=name,
            desc=desc,
            comment=comment
        )


class InitialiseGround(Task):
    """Initialise the ground state with a given thermal distribution."""

    def __init__(
            self,
            name: str = "init_ground",
            desc: str = " ",
            comment: str = " ",
            init_temp: Quantity = None
    ):
        super().__init__(
            name=name,
            desc=desc,
            comment=comment
        )
        self.params['init_temp'] = init_temp

    def initialise(self, drift_H, lindbladian=False, init_temp=None):
        """
        Prepare the initial state of the system. At the moment finite temperature requires open system dynamics.

        Parameters
        ----------
        drift_H : tf.Tensor
            Drift Hamiltonian.
        lindbladian : boolean
            Whether to include open system dynamics. Required for Temperature > 0.
        init_temp : Quantity
            Temperature of the device.

        Returns
        -------
        tf.Tensor
            State or density vector
        """
        if init_temp is None:
            init_temp = tf.cast(
                self.params['init_temp'].get_value(), dtype=tf.complex128
            )
        diag = tf.linalg.diag_part(drift_H)
        dim = len(diag)
        if abs(init_temp) > np.finfo(float).eps:  # this checks that it's not zero
            # TODO Deal with dressed basis for thermal state
            freq_diff = diag - diag[0]
            beta = 1 / (init_temp * constants.kb)
            det_bal = tf.exp(-constants.hbar * freq_diff * beta)
            norm_bal = det_bal / tf.reduce_sum(det_bal)
            dm = tf.linalg.diag(norm_bal)
            if lindbladian:
                return tf_utils.tf_dm_to_vec(dm)
            else:
                raise Warning(
                    "C3:WARNING: We still need to do Von Neumann right."
                )
        else:
            state = tf.constant(
                qt_utils.basis(dim, 0),
                shape=[dim, 1],
                dtype=tf.complex128
            )
            if lindbladian:
                return tf_utils.tf_dm_to_vec(tf_utils.tf_state_to_dm(state))
            else:
                return state


class ConfusionMatrix(Task):
    """Allows for misclassificaiton of readout measurement."""

    def __init__(
            self,
            name: str = "conf_matrix",
            desc: str = " ",
            comment: str = " ",
            **confusion_rows
    ):
        super().__init__(
            name=name,
            desc=desc,
            comment=comment
        )
        for qubit, conf_row in confusion_rows.items():
            self.params['confusion_row_'+qubit] = conf_row

    def confuse(self, pops):
        """
        Apply the confusion (or misclassification) matrix to populations.

        Parameters
        ----------
        pops : list
            Populations

        Returns
        -------
        list
            Populations after misclassification.

        """
        conf_matrix = tf.constant([[1]], dtype=tf.float64)
        for conf_row in self.params.values():
            row1 = conf_row.get_value()
            row2 = tf.ones_like(row1) - row1
            conf_mat = tf.concat([[row1], [row2]], 0)
            conf_matrix = tf_utils.tf_kron(conf_matrix, conf_mat)
        pops = tf.linalg.matmul(conf_matrix, pops)
        return pops


class MeasurementRescale(Task):
    """
    Rescale the result of the measurements.
    This is usually done to account for preparation errors.

    Parameters
    ----------
    meas_offset : Quantity
        Offset added to the measured signal.
    meas_scale : Quantity
        Factor multiplied to the measured signal.
    """

    def __init__(
            self,
            name: str = "meas_rescale",
            desc: str = " ",
            comment: str = " ",
            meas_offset: Quantity = None,
            meas_scale: Quantity = None,
    ):
        super().__init__(
            name=name,
            desc=desc,
            comment=comment
        )
        self.params['meas_offset'] = meas_offset
        self.params['meas_scale'] = meas_scale

    def rescale(self, pop1):
        """
        Apply linear rescaling and offset to the readout value.

        Parameters
        ----------
        pop1 : tf.float64
            Population in first excited state.

        Returns
        -------
        tf.float64
            Population after rescaling.
        """
        return pop1 * self.params['meas_scale'].get_value() + self.params['meas_offset'].get_value()
