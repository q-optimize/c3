"""Component class and subclasses for the components making up the quantum device."""

import numpy as np
import tensorflow as tf

from c3.c3objs import C3obj, Quantity
from c3.libraries.constants import kb, hbar
from c3.libraries.hamiltonians import hamiltonians
from c3.utils.qt_utils import hilbert_space_kron as hskron
from scipy.optimize import fmin
import tensorflow_probability as tfp

device_lib = dict()


def dev_reg_deco(func):
    """
    Decorator for making registry of functions
    """
    device_lib[str(func.__name__)] = func
    return func


class PhysicalComponent(C3obj):
    """
    Represents the components making up a chip.

    Parameters
    ----------
    hilbert_dim : int
        Dimension of the Hilbert space of this component

    """

    def __init__(self, **props):
        self.params = {}
        self.hilbert_dim = props.pop("hilbert_dim", None)
        super().__init__(**props)
        self.Hs = {}
        self.collapse_ops = {}
        self.drive_line = None

    def set_subspace_index(self, index):
        self.index = index

    def asdict(self) -> dict:
        params = {}
        for key, item in self.params.items():
            params[key] = item.asdict()
        return {
            "c3type": self.__class__.__name__,
            "params": params,
            "hilbert_dim": self.hilbert_dim,
        }


@dev_reg_deco
class Qubit(PhysicalComponent):
    """
    Represents the element in a chip functioning as qubit.

    Parameters
    ----------
    freq: np.float64
        frequency of the qubit
    anhar: np.float64
        anharmonicity of the qubit. defined as w01 - w12
    t1: np.float64
        t1, the time decay of the qubit due to dissipation
    t2star: np.float64
        t2star, the time decay of the qubit due to pure dephasing
    temp: np.float64
        temperature of the qubit, used to determine the Boltzmann distribution
        of energy level populations

    """

    def __init__(
            self,
            name,
            hilbert_dim,
            desc=None,
            comment=None,
            freq=None,
            anhar=None,
            t1=None,
            t2star=None,
            temp=None,
            params=None,
    ):
        # TODO Cleanup params passing and check for conflicting information
        super().__init__(
            name=name,
            desc=desc,
            comment=comment,
            hilbert_dim=hilbert_dim,
            params=params,
        )
        if freq:
            self.params["freq"] = freq
        if anhar:
            self.params["anhar"] = anhar
        if t1:
            self.params["t1"] = t1
        if t2star:
            self.params["t2star"] = t2star
        if temp:
            self.params["temp"] = temp

    def init_Hs(self, ann_oper):
        """
        Initialize the qubit Hamiltonians. If the dimension is higher than two, a
        Duffing oscillator is used.

        Parameters
        ----------
        ann_oper : np.array
            Annihilation operator in the full Hilbert space

        """
        resonator = hamiltonians["resonator"]
        self.Hs["freq"] = tf.constant(resonator(ann_oper), dtype=tf.complex128)
        if self.hilbert_dim > 2:
            duffing = hamiltonians["duffing"]
            self.Hs["anhar"] = tf.constant(duffing(ann_oper), dtype=tf.complex128)

    def get_Hamiltonian(self):
        """
        Compute the Hamiltonian. Multiplies the number operator with the frequency and
        anharmonicity with the Duffing part and returns their sum.

        Returns
        -------
        tf.Tensor
            Hamiltonian

        """
        h = tf.cast(self.params["freq"].get_value(), tf.complex128) * self.Hs["freq"]
        if self.hilbert_dim > 2:
            anhar = tf.cast(self.params["anhar"].get_value(), tf.complex128)
            h += anhar * self.Hs["anhar"]
        return h

    def init_Ls(self, ann_oper):
        """
        Initialize Lindbladian components.

        Parameters
        ----------
        ann_oper : np.array
            Annihilation operator in the full Hilbert space

        """
        self.collapse_ops["t1"] = ann_oper
        self.collapse_ops["temp"] = ann_oper.T.conj()
        self.collapse_ops["t2star"] = 2 * tf.matmul(ann_oper.T.conj(), ann_oper)

    def get_Lindbladian(self, dims):
        """
        Compute the Lindbladian, based on relaxation, dephasing constants and finite
        temperature.

        Returns
        -------
        tf.Tensor
            Hamiltonian

        """
        Ls = []
        if "t1" in self.params:
            t1 = self.params["t1"].get_value()
            gamma = (0.5 / t1) ** 0.5
            L = gamma * self.collapse_ops["t1"]
            Ls.append(L)
            if "temp" in self.params:
                if self.hilbert_dim > 2:
                    freq = self.params["freq"].get_value()
                    anhar = self.params["anhar"].get_value()
                    freq_diff = np.array(
                        [freq + n * anhar for n in range(self.hilbert_dim)]
                    )
                else:
                    freq_diff = np.array([self.params["freq"].get_value(), 0])
                beta = 1 / (self.params["temp"].get_value() * kb)
                det_bal = tf.exp(-hbar * tf.cast(freq_diff, tf.float64) * beta)
                det_bal_mat = hskron(tf.linalg.tensor_diag(det_bal), self.index, dims)
                L = gamma * tf.matmul(self.collapse_ops["temp"], det_bal_mat)
                Ls.append(L)
        if "t2star" in self.params:
            gamma = (0.5 / self.params["t2star"].get_value()) ** 0.5
            L = gamma * self.collapse_ops["t2star"]
            Ls.append(L)
        if Ls == []:
            raise Exception("No T1 or T2 provided")
        return tf.cast(sum(Ls), tf.complex128)


@dev_reg_deco
class Resonator(PhysicalComponent):
    """
    Represents the element in a chip functioning as resonator.

    Parameters
    ----------
    freq: np.float64
        frequency of the resonator

    """

    def init_Hs(self, ann_oper):
        """
        Initialize the Hamiltonian as a number operator

        Parameters
        ----------
        ann_oper : np.array
            Annihilation operator in the full Hilbert space.

        """
        self.Hs["freq"] = tf.constant(
            hamiltonians["resonator"](ann_oper), dtype=tf.complex128
        )

    def init_Ls(self, ann_oper):
        """NOT IMPLEMENTED"""
        pass

    def get_Hamiltonian(self):
        """Compute the Hamiltonian."""
        freq = tf.cast(self.params["freq"].get_value(), tf.complex128)
        return freq * self.Hs["freq"]

    def get_Lindbladian(self, dims):
        """NOT IMPLEMENTED"""
        pass


@dev_reg_deco
class Transmon(PhysicalComponent):
    """
    Represents the element in a chip functioning as tunanble transmon qubit.

    Parameters
    ----------
    freq: np.float64
        base frequency of the Transmon
    phi_0: np.float64
        half period of the phase dependant function
    phi: np.float64
        flux position

    """

    def __init__(
            self,
            name: str,
            desc: str = None,
            comment: str = None,
            hilbert_dim: int = None,
            freq: np.float64 = None,
            phi: np.float64 = None,
            phi_0: np.float64 = None,
            gamma: np.float64 = None,
            d: np.float64 = None,
            t1: np.float64 = None,
            t2star: np.float64 = None,
            temp: np.float64 = None,
            anhar: np.float64 = None,
            params=None,
    ):
        super().__init__(
            name=name,
            desc=desc,
            comment=comment,
            hilbert_dim=hilbert_dim,
            params=params,
        )
        if freq:
            self.params["freq"] = freq
        if phi:
            self.params["phi"] = phi
        if phi_0:
            self.params["phi_0"] = phi_0
        if d:
            self.params["d"] = d
        elif gamma:
            self.params["gamma"] = gamma
        if anhar:
            # Anharmonicity corresponding to the charging energy in the two-level case
            self.params["anhar"] = anhar
        if t1:
            self.params["t1"] = t1
        if t2star:
            self.params["t2star"] = t2star
        if temp:
            self.params["temp"] = temp
        if "d" not in self.params.keys() and "gamma" not in self.params.keys():
            print(
                "C3:WANING: No junction asymmetry specified, setting symmetric SQUID"
                " for tuning."
            )
            self.params["d"] = 0

    def get_factor(self):
        pi = tf.constant(np.pi, dtype=tf.float64)
        phi = tf.cast(self.params["phi"].get_value(), tf.float64)
        phi_0 = tf.cast(self.params["phi_0"].get_value(), tf.float64)
        if "d" in self.params:
            d = tf.cast(self.params["d"].get_value(), tf.float64)
        elif "gamma" in self.params:
            gamma = tf.cast(self.params["gamma"].get_value(), tf.complex128)
            d = (gamma - 1) / (gamma + 1)
        else:
            d = 0
        factor = tf.sqrt(
            tf.sqrt(
                tf.cos(pi * phi / phi_0) ** 2 + d ** 2 * tf.sin(pi * phi / phi_0) ** 2
            )
        )
        factor = tf.cast(factor, tf.complex128)
        return factor

    def get_anhar(self):
        anhar = tf.cast(self.params["anhar"].get_value(), tf.complex128)
        return anhar

    def get_freq(self):
        freq = tf.cast(self.params["freq"].get_value(), tf.complex128)
        anhar = tf.cast(self.params["anhar"].get_value(), tf.complex128)
        biased_freq = (freq - anhar) * self.get_factor() + anhar
        return biased_freq

    def init_Hs(self, ann_oper):
        resonator = hamiltonians["resonator"]
        self.Hs["freq"] = tf.constant(resonator(ann_oper), dtype=tf.complex128)
        if self.hilbert_dim > 2:
            duffing = hamiltonians["duffing"]
            self.Hs["anhar"] = tf.constant(duffing(ann_oper), dtype=tf.complex128)

    def init_Ls(self, ann_oper):
        """
        Initialize Lindbladian components.

        Parameters
        ----------
        ann_oper : np.array
            Annihilation operator in the full Hilbert space

        """
        self.collapse_ops["t1"] = ann_oper
        self.collapse_ops["temp"] = ann_oper.T.conj()
        self.collapse_ops["t2star"] = 2 * tf.matmul(ann_oper.T.conj(), ann_oper)

    def get_Hamiltonian(self):
        h = self.get_freq() * self.Hs["freq"]
        if self.hilbert_dim > 2:
            h += self.get_anhar() * self.Hs["anhar"]
        return h

    def get_Lindbladian(self, dims):
        """
        Compute the Lindbladian, based on relaxation, dephasing constants and finite temperature.

        Returns
        -------
        tf.Tensor
            Hamiltonian
        """
        Ls = []
        if "t1" in self.params:
            t1 = self.params["t1"].get_value()
            gamma = (0.5 / t1) ** 0.5
            L = gamma * self.collapse_ops["t1"]
            Ls.append(L)
            if "temp" in self.params:
                if self.params["temp"].get_value().numpy():
                    if self.hilbert_dim > 2:
                        freq_diff = np.array(
                            [
                                (
                                        self.params["freq"].get_value()
                                        + n * self.params["anhar"].get_value()
                                )
                                for n in range(self.hilbert_dim)
                            ]
                        )
                    else:
                        freq_diff = np.array([self.params["freq"].get_value(), 0])
                    beta = 1 / (self.params["temp"].get_value() * kb)
                    det_bal = tf.exp(-hbar * tf.cast(freq_diff, tf.float64) * beta)
                    det_bal_mat = hskron(
                        tf.linalg.tensor_diag(det_bal), self.index, dims
                    )
                    L = gamma * tf.matmul(self.collapse_ops["temp"], det_bal_mat)
                    Ls.append(L)
        if "t2star" in self.params:
            gamma = (0.5 / self.params["t2star"].get_value()) ** 0.5
            L = gamma * self.collapse_ops["t2star"]
            Ls.append(L)
        if Ls == []:
            raise Exception("No T1 or T2 provided")
        return tf.cast(sum(Ls), tf.complex128)


# @dev_reg_deco
# class Fluxonium(Qubit):
#     def __init__(
#             self,
#             name: str,
#             desc: str = None,
#             comment: str = None,
#             hilbert_dim: int = None,
#             calc_dim: int = None,
#             EC: Quantity = None,
#             EJ: Quantity = None,
#             EL: Quantity = None,
#             phi: Quantity = None,
#             phi_0:Quantity = None,
#             gamma: Quantity = None,
#             d: Quantity = None,
#             t1: np.float64 = None,
#             t2star: np.float64 = None,
#             temp: np.float64 = None,
#             anhar: np.float64 = None,
#             params=None,
#     ):
#         super().__init__(
#             name=name,
#             desc=desc,
#             comment=comment,
#             hilbert_dim=hilbert_dim,
#             freq=tf.math.sqrt(8 * EC * EL),
#             anhar=None,
#             t1=t1,
#             t2star=t2star,
#             temp=temp,
#             params=params,
#         )
#         if EC:
#             self.params["EC"] = EC
#         if EJ:
#             self.params["EJ"] = EJ
#         if EL:
#             self.params["EL"] = EL
#         if phi:
#             self.params["phi"] = phi
#         if phi_0:
#             self.params["phi_0"] = phi_0
#         if gamma:
#             self.params["gamma"] = gamma
#         if calc_dim:
#             self.params["calc_dim"] = calc_dim
#
#     def get_phase_variable(self):
#         ann_oper = tf.linalg.diag(tf.math.sqrt(tf.range(1,self.params["calc_dim"], dtype=tf.float64)), k=1)
#         EC = self.params["EC"].get_value()
#         EL = self.params["EL"].get_value()
#         phi_zpf = (2.0 * EC / EL) ** 0.25
#         return tf.cast(phi_zpf * (tf.transpose(ann_oper, conjugate=True) + ann_oper), tf.complex128)
#
#     def init_exponentiated_vars(self, ann_oper):
#         # TODO check if a 2Pi should be included in the exponentiation
#         self.exp_phi_op = tf.linalg.expm(1.j * self.get_phase_variable())
#
#     def get_freq(self):
#         EC = self.params["EC"].get_value()
#         EL = self.params["EL"].get_value()
#         return tf.cast(tf.math.sqrt(8.0 * EL * EC), tf.complex128)
#
#     def init_Hs(self, ann_oper):
#         self.init_exponentiated_vars(ann_oper)
#         resonator = hamiltonians["resonator"]
#         self.Hs["freq"] = tf.constant(resonator(ann_oper), dtype=tf.complex128)
#         # self.Hs["freq"] = tf.cast(tf.linalg.diag(tf.range(self.params['calc_dim'], dtype=tf.float64)), tf.complex128)
#
#     def get_Hamiltonian(self):
#         EJ = tf.cast(self.params["EJ"].get_value(), tf.complex128)
#         phi = tf.cast(self.params["phi"].get_value(), tf.complex128)
#         phi_0 = tf.cast(self.params["phi_0"].get_value(), tf.complex128)
#         phase = tf.cast(1j * 2 * np.pi * phi / phi_0, tf.complex128)
#         exp_mat = self.exp_phi_op * tf.math.exp(phase)
#         cos_mat = tf.cast(tf.math.real(exp_mat), tf.complex128)
#         # cos_mat = 0.5 * (exp_mat + tf.transpose(exp_mat, conjugate=True))
#         cos_mat = cos_mat[:self.hilbert_dim, :self.hilbert_dim]
#         h = self.get_freq() * self.Hs["freq"]
#         h -= EJ * cos_mat
#         return tf.cast(tf.math.real(h), tf.complex128) #TODO apply kronecker product

import matplotlib.pyplot as plt
@dev_reg_deco
class CShuntFluxQubitCos(Qubit):
    def __init__(
            self,
            name: str,
            desc: str = None,
            comment: str = None,
            hilbert_dim: int = None,
            calc_dim: int = None,
            EC: Quantity = None,
            EJ: Quantity = None,
            EL: Quantity = None,
            phi: Quantity = None,
            phi_0: Quantity = None,
            gamma: Quantity = None,
            d: Quantity = None,
            t1: np.float64 = None,
            t2star: np.float64 = None,
            temp: np.float64 = None,
            anhar: np.float64 = None,
            params=None,
    ):
        super().__init__(
            name=name,
            desc=desc,
            comment=comment,
            hilbert_dim=hilbert_dim,
            freq=0,  # tf.math.sqrt(8 * EC * EL),
            anhar=None,
            t1=t1,
            t2star=t2star,
            temp=temp,
            params=params,
        )
        if EC:
            self.params["EC"] = EC
        if EJ:
            self.params["EJ"] = EJ
        if EL:
            self.params["EL"] = EL
        if phi:
            self.params["phi"] = phi
        if phi_0:
            self.params["phi_0"] = phi_0
        if gamma:
            self.params["gamma"] = gamma
        if calc_dim:
            self.params["calc_dim"] = calc_dim

    def get_phase_variable(self):
        ann_oper = tf.linalg.diag(tf.math.sqrt(tf.range(1,self.params["calc_dim"], dtype=tf.float64)), k=1)
        EC = self.params["EC"].get_value()
        EJ = self.params["EJ"].get_value()
        phi_zpf = (2.0 * EC / EJ) ** 0.25
        return tf.cast(phi_zpf * (tf.transpose(ann_oper, conjugate=True) + ann_oper), tf.complex128)

    def get_n_variable(self):
        ann_oper = tf.linalg.diag(tf.math.sqrt(tf.range(1,self.params["calc_dim"], dtype=tf.float64)), k=1)
        EC = self.params["EC"].get_value()
        EJ = self.params["EJ"].get_value()
        n_zpf = (EJ / EC / 32) ** 0.25
        return tf.cast(n_zpf * (- tf.transpose(ann_oper, conjugate=True) + ann_oper), tf.complex128)

    def init_exponentiated_vars(self, ann_oper):
        # TODO check if a 2Pi should be included in the exponentiation
        self.exp_phi_op = tf.linalg.expm(1.j * self.get_phase_variable())

    def get_freq(self):
        EC = self.params["EC"].get_value()
        EL = self.params["EL"].get_value()
        return tf.cast(tf.math.sqrt(8.0 * EL * EC), tf.complex128)

    def init_Hs(self, ann_oper):
        # self.init_exponentiated_vars(ann_oper)
        # resonator = hamiltonians["resonator"]
        # self.Hs["freq"] = tf.constant(resonator(ann_oper), dtype=tf.complex128)
        # # self.Hs["freq"] = tf.cast(tf.linalg.diag(tf.range(self.params['calc_dim'], dtype=tf.float64)), tf.complex128)
        pass

    def cosm(self, var, a=1, b=0):
        exponent = 1j * (a * var)
        exp_mat = tf.linalg.expm(exponent) * tf.exp(1j * b)
        cos_mat = 0.5 * (exp_mat + tf.transpose(exp_mat, conjugate=True))
        # plt.imshow(tf.math.imag(cos_mat))
        # plt.colorbar()
        # plt.show()
        # plt.imshow(tf.math.real(cos_mat))
        # plt.colorbar()
        # plt.show()
        return cos_mat

    def get_Hamiltonian(self):
        EJ = tf.cast(self.params["EJ"].get_value(), tf.complex128)
        EC = tf.cast(self.params["EC"].get_value(), tf.complex128)
        gamma = tf.cast(self.params["gamma"].get_value(), tf.complex128)
        phi = tf.cast(self.params["phi"].get_value(), tf.complex128)
        phi_0 = tf.cast(self.params["phi_0"].get_value(), tf.complex128)
        phase = tf.cast(2 * np.pi * phi / phi_0, tf.complex128)
        phi_variable = self.get_phase_variable()
        n = self.get_n_variable()
        h = 4 * EC * n + EJ * (-  1 * self.cosm(phi_variable, 2, phase) - 2 * gamma * self.cosm(phi_variable))

        return tf.cast(tf.math.real(h), tf.complex128) #TODO apply kronecker product

@dev_reg_deco
class CShuntFluxQubit(Qubit):
    def __init__(
            self,
            name: str,
            desc: str = None,
            comment: str = None,
            hilbert_dim: int = None,
            calc_dim: int = None,
            EC: Quantity = None,
            EJ: Quantity = None,
            EL: Quantity = None,
            phi: Quantity = None,
            phi_0: Quantity = None,
            gamma: Quantity = None,
            d: Quantity = None,
            t1: np.float64 = None,
            t2star: np.float64 = None,
            temp: np.float64 = None,
            anhar: np.float64 = None,
            params=dict(),
            resolution=None
    ):
        super().__init__(
            name=name,
            desc=desc,
            comment=comment,
            hilbert_dim=hilbert_dim,
            freq=0,  # tf.math.sqrt(8 * EC * EL),
            anhar=None,
            t1=t1,
            t2star=t2star,
            temp=temp,
            params=params,
        )

        self.inputs = params.pop("inputs", 1)
        self.outputs = params.pop("outputs", 0)
        if resolution:
            self.resolution = resolution
            self.inputs = 1
            self.outputs = 2
        if EC:
            self.params["EC"] = EC
        if EJ:
            self.params["EJ"] = EJ
        if EL:
            self.params["EL"] = EL
        if phi:
            self.params["phi"] = phi
        if phi_0:
            self.params["phi_0"] = phi_0
        if gamma:
            self.params["gamma"] = gamma
        if calc_dim:
            self.params["calc_dim"] = calc_dim

        self.phi_var_min_ref = None
        self.min_phi_var_change_test = 1312341234 # Random Wrong Number

    def get_potential_function(self, phi_variable, deriv_order=1, phi_sig=0):
        phi = (self.params["phi"].get_value() + phi_sig) / self.params["phi_0"].get_value() * 2 * np.pi
        gamma = self.params["gamma"].get_value()
        EJ = self.params["EJ"].get_value()
        phi_variable = tf.cast(phi_variable, tf.float64)
        if deriv_order == 0:  # Has to be defined
            return EJ * (-  1 * tf.cos(phi_variable + phi) - 2 * gamma * tf.cos(phi_variable / 2))
        elif deriv_order == 1:
            return EJ * (+  2 * tf.sin(phi_variable + phi) + 2 * gamma * tf.sin(phi_variable / 2)) /2 # TODO: Why is this different than Krantz????
        elif deriv_order == 2:
            return EJ * (+  4 * tf.cos(phi_variable + phi) + 2 * gamma * tf.cos(phi_variable / 2)) / 4
        elif deriv_order == 3:
            return EJ * (-  8 * tf.sin(phi_variable + phi) - 2 * gamma * tf.sin(phi_variable / 2)) / 8
        elif deriv_order == 4:
            return EJ * (- 16 * tf.cos(phi_variable + phi) - 2 * gamma * tf.cos(phi_variable / 2)) / 16
        else:  # Calculate derivative by tensorflow
            with tf.GradientTape() as tape:
                tape.watch(phi_variable)
                val = self.get_potential_function(phi_variable, deriv_order - 1)
            return tape.gradient(val, phi_variable)

    def get_minimum_phi_var(self, init_phi_variable: tf.float64 = 0, phi_sig=0):
        # TODO maybe improve to analytical funciton here
        # TODO do not reevaluate if not necessary
        phi_0 = self.params["phi_0"].get_value()
        initial_pot_eval = self.get_potential_function(0., 0)
        if self.min_phi_var_change_test != initial_pot_eval and phi_sig == 0:
            phi_var_min = fmin(self.get_potential_function, [init_phi_variable], args=(0, 0), disp=False)
            self.min_phi_var_interpolation = None
            self.min_phi_var_change_test = initial_pot_eval
            print(phi_var_min)
            return phi_var_min
        if not(self.phi_var_min_ref is not None and self.min_phi_var_change_test == initial_pot_eval):
            print(self.params["phi"], phi_sig)
            phi_var_min_ref = list()
            print('a')
            for phi_i in np.linspace(0, phi_0, 50):  # Interpolate over 50 points
                phi_var_min_ref.append(fmin(self.get_potential_function, [init_phi_variable], args=(0, phi_i), disp=False))
            print('b')
            self.phi_var_min_ref = tf.reshape(tf.constant(phi_var_min_ref, tf.float64), len(phi_var_min_ref))
            # self.min_phi_var_interpolation = lambda x: tfp.math.interp_regular_1d_grid(tf.math.mod(x, phi_0), 0, phi_0, phi_var_min_ref)
            self.min_phi_var_change_test = initial_pot_eval

        phi_var_min = tfp.math.interp_regular_1d_grid(tf.math.mod(phi_sig, phi_0), 0, phi_0, self.phi_var_min_ref)
        return phi_var_min

        # gamma = self.params["gamma"].get_value()
        # return tf.cast(0.5, tf.float64)

    def get_frequency(self, phi_sig=0):
        EC = self.params["EC"].get_value()
        EJ = self.params["EJ"].get_value()
        phi_var_min = self.get_minimum_phi_var(phi_sig=phi_sig)
        second_order_deriv = self.get_potential_function(phi_var_min, 2, phi_sig=phi_sig)
        fourth_order_deriv = self.get_potential_function(phi_var_min, 4, phi_sig=phi_sig)
        # if type(phi_sig) is not int:
        #     print(phi_var_min.shape, phi_sig.shape, second_order_deriv.shape)
        return tf.math.sqrt(2 * EJ * EC) + tf.math.sqrt(2 * EC / EJ) * second_order_deriv + EC / EJ * fourth_order_deriv

    get_freq = get_frequency

    def get_anharmonicity(self, phi_sig=0):
        EC = self.params["EC"].get_value()
        EJ = self.params["EJ"].get_value()
        phi_var_min = self.get_minimum_phi_var()
        fourth_order_deriv = self.get_potential_function(phi_var_min, 4, phi_sig=phi_sig)
        return EC / EJ * fourth_order_deriv

    def get_third_order_prefactor(self, phi_sig=0):
        EC = self.params["EC"].get_value()
        EJ = self.params["EJ"].get_value()
        phi_var_min = self.get_minimum_phi_var()
        third_order_deriv = self.get_potential_function(phi_var_min, 3, phi_sig=phi_sig)
        return 0.5 * ((2 * EC / EJ) ** 0.75) * third_order_deriv

    def init_Hs(self, ann_oper):
        """
        initialize Hamiltonians for cubic hamiltinian
        Parameters
        ----------
        ann_oper : np.array
            Annihilation operator in the full Hilbert space
        """
        resonator = hamiltonians["resonator"]
        self.Hs["freq"] = tf.constant(resonator(ann_oper), dtype=tf.complex128)
        if self.hilbert_dim > 2:
            duffing = hamiltonians["duffing"]
            self.Hs["anhar"] = tf.constant(duffing(ann_oper), dtype=tf.complex128)
        third = hamiltonians["third_order"]
        self.Hs["third_order"] = tf.constant(third(ann_oper), dtype=tf.complex128)
        self.signal_h = None

    def get_Hamiltonian(self) -> tf.Tensor:
        """
        Calculate the hamiltonian
        Returns
        -------
        tf.Tensor
            Hamiltonian
        """
        h = tf.cast(self.get_frequency(), tf.complex128) * self.Hs["freq"]
        # h += tf.cast(self.get_third_order_prefactor(), tf.complex128) * self.Hs["third_order"]
        if self.hilbert_dim > 2:
            h += tf.cast(self.get_anharmonicity(), tf.complex128) * self.Hs["anhar"]
        return h

    def process(self, instr, chan: str, signal_in):
        sig = signal_in['values']
        anharmonicity = self.get_anharmonicity(sig)
        frequency = self.get_frequency(sig)
        third_order = self.get_third_order_prefactor(sig)
        h = tf.expand_dims(tf.expand_dims(tf.cast(frequency, tf.complex128), 1), 2) * self.Hs["freq"]
        if self.hilbert_dim > 2:
            # h += tf.expand_dims(tf.expand_dims(tf.cast(third_order, tf.complex128), 1), 2) * self.Hs["third_order"]
            h += tf.expand_dims(tf.expand_dims(tf.cast(anharmonicity, tf.complex128), 1), 2) * self.Hs["anhar"]
        self.signal_h = h
        return {"ts": signal_in["ts"], "frequency":frequency, "anharmonicity": anharmonicity}#, "#third order": third_order}


@dev_reg_deco
class Fluxonium(CShuntFluxQubit):
    def __init__(
            self,
            name: str,
            desc: str = None,
            comment: str = None,
            hilbert_dim: int = None,
            calc_dim: int = None,
            EC: Quantity = None,
            EJ: Quantity = None,
            EL: Quantity = None,
            phi: Quantity = None,
            phi_0: Quantity = None,
            gamma: Quantity = None,
            t1: np.float64 = None,
            t2star: np.float64 = None,
            temp: np.float64 = None,
            params=None,
    ):
        super().__init__(name=name,
                         desc=desc,
                         comment=comment,
                         hilbert_dim=hilbert_dim,
                         EC=EC,
                         EJ=EJ,
                         phi=phi,
                         phi_0=phi_0,
                         gamma=gamma,
                         t1=t1,
                         t2star=t2star,
                         temp=temp,
                         params=params)
        # if EC:
        #     self.params["EC"] = EC
        # if EJ:
        #     self.params["EJ"] = EJ
        if EL:
            self.params["EL"] = EL
        # if phi:
        #     self.params["phi"] = phi
        # if phi_0:
        #     self.params["phi_0"] = phi_0
        # if gamma:
        #     self.params["gamma"] = gamma
        # if calc_dim:
        #     self.params["calc_dim"] = calc_dim

    def get_potential_function(self, phi_variable, deriv_order=1) -> tf.float64:
        EL = self.params["EL"].get_value()
        EJ = self.params["EJ"].get_value()
        phi = self.params["phi"].get_value() / self.params["phi_0"].get_value() * 2 * np.pi
        if deriv_order == 0:  # Has to be defined
            return -EJ * tf.math.cos(phi_variable + phi) + 0.5 * EL * phi_variable ** 2
        elif deriv_order == 1:
            return  EJ * tf.math.sin(phi_variable + phi) + EL * phi_variable
        elif deriv_order == 2:
            return  EJ * tf.math.cos(phi_variable + phi) + EL
        elif deriv_order == 3:
            return  - EJ * tf.math.sin(phi_variable + phi)
        else:  # Calculate derivative by tensorflow
            with tf.GradientTape() as tape:
                tape.watch(phi_variable)
                val = self.get_potential_function(phi_variable, deriv_order - 1)
            grad = tape.gradient(val, phi_variable)
            return grad
    #
    # def get_minimum_phi_var(self, init_phi_variable: tf.float64 = 0) -> tf.float64:
    #     # Redefine here as minimizing function does not work otherwise
    #     # TODO maybe improve to analytical funciton here
    #     # TODO do not reevaluate if not necessary
    #     phi_var_min = fmin(self.get_potential_function, [init_phi_variable], args=(0,), disp=False)
    #     return tf.constant(phi_var_min, tf.float64)
    #
    # def part(self):
    #     pass


@dev_reg_deco
class SNAIL(Qubit):
    """
    Represents the element in a chip functioning as a three wave mixing element also knwon as a SNAIL.
    Reference: https://arxiv.org/pdf/1702.00869.pdf
    Parameters
    ----------
    freq: np.float64
        frequency of the qubit
    anhar: np.float64
        anharmonicity of the qubit. defined as w01 - w12
    beta: np.float64
        third order non_linearity of the qubit.
    t1: np.float64
        t1, the time decay of the qubit due to dissipation
    t2star: np.float64
        t2star, the time decay of the qubit due to pure dephasing
    temp: np.float64
        temperature of the qubit, used to determine the Boltzmann distribution
        of energy level populations
    Class is mostly an exact copy of the Qubit class. The only difference is the added third order non linearity with a prefactor beta.
    The only modification is the get hamiltonian and init hamiltonian definition. Also imported the necessary third order non linearity
    from the hamiltonian library.
    """

    def __init__(
            self,
            name: str,
            desc: str = " ",
            comment: str = " ",
            hilbert_dim: int = 4,
            freq: np.float64 = None,
            anhar: np.float64 = None,
            beta: np.float64 = None,
            t1: np.float64 = None,
            t2star: np.float64 = None,
            temp: np.float64 = None,
            params: dict = None
    ):
        super().__init__(
            name=name,
            desc=desc,
            comment=comment,
            hilbert_dim=hilbert_dim,
            t1=t1,
            t2star=t2star,
            temp=temp,
            params=params,
        )
        self.params["freq"] = freq
        self.params["beta"] = beta
        if anhar:
            self.params["anhar"] = anhar
        if t1:
            self.params["t1"] = t1
        if t2star:
            self.params["t2star"] = t2star
        if temp:
            self.params["temp"] = temp

    def init_Hs(self, ann_oper):
        """
        Initialize the SNAIL Hamiltonians.
        Parameters
        ----------
        ann_oper : np.array
            Annihilation operator in the full Hilbert space
        """
        resonator = hamiltonians["resonator"]
        self.Hs["freq"] = tf.constant(resonator(ann_oper), dtype=tf.complex128)
        if self.hilbert_dim > 2:
            duffing = hamiltonians["duffing"]
            self.Hs["anhar"] = tf.constant(duffing(ann_oper), dtype=tf.complex128)
        third = hamiltonians["third_order"]
        self.Hs["beta"] = tf.constant(third(ann_oper), dtype=tf.complex128)

    def get_Hamiltonian(self):
        """
        Compute the Hamiltonian. Multiplies the number operator with the frequency and anharmonicity with
        the Duffing part and returns their sum.
        Returns
        -------
        tf.Tensor
            Hamiltonian
        """
        h = tf.cast(self.params["freq"].get_value(), tf.complex128) * self.Hs["freq"]
        h += tf.cast(self.params["beta"].get_value(), tf.complex128) * self.Hs["beta"]
        if self.hilbert_dim > 2:
            h += (
                    tf.cast(self.params["anhar"].get_value(), tf.complex128)
                    * self.Hs["anhar"]
            )

        return h


@dev_reg_deco
class LineComponent(C3obj):
    """
    Represents the components connecting chip elements and drives.

    Parameters
    ----------
    connected: list
        specifies the component that are connected with this line

    """

    def __init__(self, **props):
        h_func = props.pop("hamiltonian_func")
        self.connected = props.pop("connected")
        if callable(h_func):
            self.hamiltonian_func = h_func
        else:
            self.hamiltonian_func = hamiltonians[h_func]
        super().__init__(**props)
        self.Hs = {}

    def asdict(self) -> dict:
        params = {}
        for key, item in self.params.items():
            params[key] = item.asdict()
        return {
            "c3type": self.__class__.__name__,
            "params": params,
            "hamiltonian_func": self.hamiltonian_func.__name__,
            "connected": self.connected,
        }


@dev_reg_deco
class Coupling(LineComponent):
    """
    Represents a coupling behaviour between elements.

    Parameters
    ----------
    strength: np.float64
        coupling strength
    connected: list
        all physical components coupled via this specific coupling

    """

    def __init__(
            self,
            name,
            desc=None,
            comment=None,
            strength=None,
            connected=None,
            params=None,
            hamiltonian_func=None,
    ):
        super().__init__(
            name=name,
            desc=desc,
            comment=comment,
            params=params,
            connected=connected,
            hamiltonian_func=hamiltonian_func,
        )
        if strength:
            self.params["strength"] = strength

    def init_Hs(self, opers_list):
        self.Hs["strength"] = tf.constant(
            self.hamiltonian_func(opers_list), dtype=tf.complex128
        )

    def get_Hamiltonian(self):
        strength = tf.cast(self.params["strength"].get_value(), tf.complex128)
        return strength * self.Hs["strength"]


@dev_reg_deco
class Drive(LineComponent):
    """
    Represents a drive line.

    Parameters
    ----------
    connected: list
        all physical components receiving driving signals via this line

    """

    def init_Hs(self, ann_opers: list):
        hs = []
        for a in ann_opers:
            hs.append(tf.constant(self.hamiltonian_func(a), dtype=tf.complex128))
        self.h = sum(hs)

    def get_Hamiltonian(self):
        return self.h
