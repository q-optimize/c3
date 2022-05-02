"""Module to implement a C3Options class derived from the native
qiskit Options class"""

from typing import Any, Dict
from qiskit.providers import Options


class C3Options(Options):
    """A derived class of the base Options class in Qiskit.

    It only provides an additional to_dict() method to obtain
    all the options as a single dictionary

    Parameters
    ----------
    Options : qiskit.providers.Options
        Base class for Options. The get() and update_options() are
        used from this base class
    """

    def to_dict(self) -> Dict[str, Any]:
        """Return a dict representation of the options

        Returns
        -------
        Dict[str, Any]
            The Options in the form of a dictionary
        """
        out_dict = {
            "params": self.params,
            "opt_map": self.opt_map,
            "shots": self.shots,
            "memory": self.memory,
            "initial_statevector": self.initial_statevector,
        }
        return out_dict
