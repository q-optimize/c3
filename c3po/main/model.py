import qutip as qt
import tensorflow as tf

from c3po import utils


class Model:
    """
    What the theorist thinks about from the system.
    Class to store information about our system/problem/device. Different
    models can represent the same system.

    Parameters
    ---------
    component_parameters : dict of dict
    couplings : dict of dict
    hilbert_space : dict
        Hilbert space dimensions of full space
    comp_hilbert_space : dict
        Hilbert space dimensions of computational space

    Attributes
    ----------
    H0: :class:'qutip.qobj' Drift Hamiltonian
    Hcs: :class:'list of qutip.qobj' Control Hamiltonians
    H_tf : empty, constructed when needed

    component_parameters :

    control_fields: list
        [args, func1_t, func2_t, ,...]

    coupling :

    hilbert_space :

    Methods
    -------
    construct_Hamiltonian(component_parameters, hilbert_space)
        Construct a model for this system, to be used in numerics.
    get_Hamiltonian()
        Returns the Hamiltonian in a QuTip compatible way
    get_time_slices()
    """
    def __init__(self, component_parameters, coupling, hilbert_space,tf_flag="False"):

        hbar = 1

        self.component_parameters = component_parameters
        self.coupling = coupling
        self.hilbert_space = hilbert_space
        self.Hcs = []

        omega_q = component_parameters['qubit_1']['freq']
        omega_r = component_parameters['cavity']['freq']
        g = coupling['q1_cav']['strength']

        dim_q = hilbert_space['qubit_1']
        dim_r = hilbert_space['cavity']

        a = qt.tensor(qt.qeye(dim_q), qt.destroy(dim_r))
        sigmaz = qt.tensor(qt.sigmaz(), qt.qeye(dim_r))
        sigmax = qt.tensor(qt.sigmax(), qt.qeye(dim_r))

        self.H0 = hbar * omega_q / 2 * sigmaz + hbar * omega_r * a.dag() * a \
            + hbar * g * (a.dag() + a) * sigmax
        H1 = hbar * sigmax
        self.Hcs.append(H1)

        if tf_flag == "True":
            print("initialize tensorflow parts of model")
            self.init_tf_model()


    def init_tf_model(self):
        tf_hbar = tf.constant(1, dtype=tf.float64, name='planck')

        self.tf_Hcs = []

        # should name of placeholder become concat of keys, e.g.
        # name="qubit_1/freq"? This might be helpfull to retrieve parameter
        # information
        # tf_omega_q = tf.placeholder(tf.complex64, shape=(), name="Qubit_frequency")
        # tf_omega_r = tf.placeholder(tf.complex64, shape=(), name="Cavity_frequency")
        # tf_g = tf.placeholder(tf.complex64, shape=(), name="Coupling")

        omega_q = self.component_parameters['qubit_1']['freq']
        omega_r = self.component_parameters['cavity']['freq']
        g = self.coupling['q1_cav']['strength']

        tf_omega_q = tf.constant(
            omega_q,
            dtype=tf.float64,
            name="Qubit_frequency"
            )
        tf_omega_r = tf.constant(
            omega_r,
            dtype=tf.float64,
            name="Cavity_frequency"
            )
        tf_g = tf.constant(g, dtype=tf.float64, name="Coupling")

        dim_q = self.hilbert_space['qubit_1']
        dim_r = self.hilbert_space['cavity']

        a = qt.tensor(qt.qeye(dim_q), qt.destroy(dim_r))
        a_dag = a.dag()

        tf_a = tf.constant(a.full(), dtype=tf.complex128, name="a")
        tf_a_dag = tf.constant(a_dag.full(), dtype=tf.complex128, name="a_dag")
        tf_sigmaz = tf.constant(
            qt.tensor(qt.sigmaz(), qt.qeye(dim_r)).full(), dtype=tf.complex128,
             name="sigma_z"
             )
        tf_sigmax = tf.constant(
            qt.tensor(qt.sigmax(), qt.qeye(dim_r)).full(),
            dtype=tf.complex128,
            name="sigma_x"
            )

        with tf.name_scope('drift_hamiltonian'):
            self.tf_H0 = tf.cast(tf_hbar, tf.complex128) * tf.cast((tf_omega_q / 2), tf.complex128) * tf_sigmaz \
                        + tf.cast(tf_hbar, tf.complex128) * tf.cast(tf_omega_r, tf.complex128) * tf_a_dag * tf_a \
                        + tf.cast((tf_hbar * tf_g), tf.complex128) * (tf_a_dag + tf_a) * tf_sigmax

        with tf.name_scope('control_hamiltonian'):
            tf_H1 = tf.cast(tf_hbar, tf.complex128) * tf_sigmax

        self.tf_Hcs.append(tf_H1)

    # Is the session needed by model (aka does the session need to be passed
    # down in the code?)
    # should the session setup be part of model init? if so is this function
    # obsolete ?
    def set_tf_session(self, tf_session):
        self.tf_session = tf_session

    # TODO: Think about the distinction between System and Model classes
    """
    Federico: I believe the information about the physical system,
    i.e. components and companent parameters should be in the system class
    Then the Hamiltonian is constructed in the model class with parsers
    (as above) or just provided by the user
    """

    def get_Hamiltonian(self, control_fields):
        H = [self.H0]
        for ii in range(len(control_fields)):
            H.append([self.Hcs[ii], control_fields[ii]])
        return H

    def get_tf_Hamiltonian(self, control_fields):
        tf_H = [self.tf_H0]
        for ii in range(len(control_fields)):
            tf_H.append([self.tf_Hcs[ii], control_fields[ii]])
        return tf_H
