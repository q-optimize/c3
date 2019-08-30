import qutip as qt
import tensorflow as tf
from c3po import utils
from c3po.cobj.component import *

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


    def __init__(
            self,
            chip_elements
            ):

        self.control_Hs = []
        self.drift_Hs = []

        # Construct array with dimension of elements (only qubits & resonators)
        self.dims = []
        self.names = []
        for element in chip_elements:
            if isinstance(element, Qubit) or isinstance(element, Resonator):
                self.dims.append(element.hilbert_dim)
                self.names.append(element.name)

        # Create anninhilation operators for physical elements
        self.ann_opers = []
        for indx in range(len(self.dims)):
            a = qt.destroy(self.dims[indx])
            for indy in range(len(self.dims)):
                qI = qt.qeye(self.dims[indy])
                if indy < indx:
                    a = qt.tensor(qI, a)
                if indy > indx:
                    a = qt.tensor(a, qI)
            self.ann_opers.append(a)

        # Create drift Hamiltonian matrices
        self.drift_Hs = []
        for indx in range(len(chip_elements)):
            element = chip_elements[indx]
            if isinstance(element, Qubit) or isinstance(element, Resonator):
                el_indx = self.names.index(element.name)
                ann_oper = self.ann_opers[el_indx]
                self.drift_Hs.append(element.get_hamiltonian(ann_oper))
            elif isinstance(element, Coupling):
                el_indxs = []
                for connected_element in element.connected:
                    el_indxs.append(self.names.index(connected_element))
                ann_opers = [self.ann_opers[el_indx] for el_indx in el_indxs]
                self.drift_Hs.append(element.get_hamiltonian(ann_opers))
            elif isinstance(element, Drive):
                el_indxs = []
                for connected_element in element.connected:
                    el_indxs.append(self.names.index(connected_element))
                ann_opers = [self.ann_opers[el_indx] for el_indx in el_indxs]
                self.control_Hs.append(element.get_hamiltonian(ann_opers))


    def get_Hamiltonians(self):
        H0 = sum(self.drift_Hs)
        drift_H = tf.constant(H0.full(), dtype=tf.complex128, name="H_drift")
        control_Hs = []
        for ctrl_H in self.control_Hs:
            hc =  tf.constant(
                ctrl_H.full(),
                dtype=tf.complex128,
                name="hc")
            control_Hs.append(hc)
        return drift_H, control_Hs
