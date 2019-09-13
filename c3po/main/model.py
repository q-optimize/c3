import numpy as np
import qutip as qt
import tensorflow as tf
from c3po.utils.hamiltonians import *
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

        self.chip_elements = chip_elements
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

            self.ann_opers.append(
                tf.constant(a.full(), dtype=tf.complex128)
                )

        # Create drift Hamiltonian matrices and model parameter vector
        self.params = []
        self.params_desc = []
        self.drift_Hs = []
        for element in chip_elements:

            if isinstance(element, Qubit) or isinstance(element, Resonator):
                el_indx = self.names.index(element.name)
                ann_oper = self.ann_opers[el_indx]

                self.drift_Hs.append(resonator(ann_oper))
                self.params.append(element.values['freq'])
                self.params_desc.append([element.name, 'freq'])

                if isinstance(element, Qubit):
                    self.drift_Hs.append(duffing(ann_oper))
                    self.params.append(element.values['delta'])
                    self.params_desc.append([element.name, 'delta'])

            elif isinstance(element, Coupling):
                el_indxs = []

                for connected_element in element.connected:
                    el_indxs.append(self.names.index(connected_element))

                ann_opers = [self.ann_opers[el_indx] for el_indx in el_indxs]

                self.drift_Hs.append(int_XX(ann_opers))
                self.params.append(element.values['strength'])
                self.params_desc.append([element.name, 'strength'])

            elif isinstance(element, Drive):
                el_indxs = []

                for connected_element in element.connected:
                    el_indxs.append(self.names.index(connected_element))

                ann_opers = [self.ann_opers[el_indx] for el_indx in el_indxs]
                self.control_Hs.append(drive(ann_opers))

        self.params = np.array(self.params)


    def update_parameters(self, new_params):
        idx = 0
        self.params = new_params
        for element in self.chip_elements:

            if isinstance(element, Qubit):
                element.values['freq'] = new_params[idx]
                idx += 1
                element.values['delta'] = new_params[idx]
                idx += 1

            elif isinstance(element, Resonator):
                element.values['freq'] = new_params[idx]
                idx += 1

            elif isinstance(element, Coupling):
                element.values['strength'] = new_params[idx]
                idx += 1


    def get_Hamiltonians(self, params=None):
        if params is None:
            params = self.params

        drift_H = tf.zeros_like(self.drift_Hs[0])

        for ii in range(len(self.drift_Hs)):
            drift_H += tf.cast(self.params[ii], tf.complex128) * self.drift_Hs[ii]

        return drift_H, self.control_Hs


    def get_values_bounds(self):
        values = self.params
        bounds = [0.5*self.params, 1.5*self.params]
        return values, bounds
