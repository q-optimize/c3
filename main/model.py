import qutip as qt
import tensorflow as tf



# TODO: Think about the distinction between System and Model classes


class Model:
    """
    What the theorist thinks about from the system.
    Class to store information about our system/problem/device. Different
    models can represent the same system.

    Parameters
    ---------
    physical_parameters : dict
        Represents the beta in GOAT language. Contains physical parameters as
        well as Hilbert space dimensions, bounds
    hilbert_space : dict
        Hilbert space dimensions of computational and full spaces

    Attributes
    ----------
    H: :class:'qutip.qobj' System hamiltonian or a list of Drift and Control
        terms
    H_tf : empty, constructed when needed

    component_parameters :

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
        tf_hbar = tf.constant(1, tf.complex64, name='planck')

        self.tf_Hcs = []

        # should name of placeholder become concat of keys, e.g. 
        # name="qubit_1/freq"? This might be helpfull to retrieve parameter
        # information 
        tf_omega_q = tf.placeholder(tf.complex64, shape=(), name="Resonator_frequency")
        tf_omega_r = tf.placeholder(tf.complex64, shape=(), name="Qubit_frequency")
        tf_g = tf.placeholder(tf.complex64, shape=(), name="Coupling")

        dim_q = self.hilbert_space['qubit_1']
        dim_r = self.hilbert_space['cavity']

        a = qt.tensor(qt.qeye(dim_q), qt.destroy(dim_r))
        a_dag = a.dag()

        tf_a = tf.constant(a.full(), tf.complex64, name="a")
        tf_a_dag = tf.constant(a_dag.full(), tf.complex64, name="a_dag")
        tf_sigmaz = tf.constant(qt.tensor(qt.sigmaz(), qt.qeye(dim_r)).full(), tf.complex64, name="sigma_z")
        tf_sigmax = tf.constant(qt.tensor(qt.sigmax(), qt.qeye(dim_r)).full(), tf.complex64, name="sigma_x")

        with tf.name_scope('drift_hamiltonian'):
            self.tf_H0 = tf_hbar * tf_omega_q / 2 * tf_sigmaz \
                        + tf_hbar * tf_omega_r * tf_a_dag * tf_a \
                        + tf_hbar * tf_g * (tf_a_dag + tf_a) * tf_sigmax

        with tf.name_scope('control_hamiltonian'):
            tf_H1 = tf_hbar * tf_sigmax

        self.tf_Hcs.append(tf_H1)

    # Is the session needed by model (aka does the session need to be passed 
    # down in the code?)
    # should the session setup be part of model init? if so is this function
    # obsolete ?
    def set_tf_session(self, tf_session):
        self.tf_session = tf_session

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


