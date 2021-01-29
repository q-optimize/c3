from collections import OrderedDict
from qiskit.exceptions import QiskitError
from qiskit.providers.provider import ProviderV1
from qiskit.providers.exceptions import QiskitBackendNotFoundError

from .c3_backend import C3QasmSimulator

SIMULATORS = [C3QasmSimulator]


class C3Provider(ProviderV1):
    """Provider for C3 Qiskit backends

    Parameters
    ----------
    ProviderV1 : ProviderV1
        Derived from ProviderV1 from qiskit.providers.provider
    """

    def __init__(self):
        super().__init__()

        self.name = "c3_provider"
        self._backends = self._verify_backends()

    def backends(self, name=None, filters=None, **kwargs):
        """Return a list of backends matching the name

        Parameters
        ----------
        name : str, optional
            name of the backend, by default None
        filters : callable, optional
            Filtering conditions, as callable, by default None

        Returns
        -------
        list[BackendV1]
            A list of backend instances matching the condition
        """
        backends = list(self._backends.values())
        if name:
            try:
                backends = [self._backends[name]]
            except LookupError:
                raise QiskitBackendNotFoundError(
                    "The '{}' backend is not installed in your system.".format(name)
                )

        return backends

    def _verify_backends(self):
        """Return instantiated Backends

        Returns
        -------
        dict[str:BackendV1]
            A dict of the instantiated backends keyed by backend name
        """

        ret = OrderedDict()
        for backend_cls in SIMULATORS:
            backend_instance = self._get_backend_instance(backend_cls)
            backend_name = backend_instance.name()
            ret[backend_name] = backend_instance
        return ret

    def _get_backend_instance(self, backend_cls):
        """Return an instance of a backend from its class

        Parameters
        ----------
        backend_cls : class
            backend class

        Returns
        -------
        BackendV1
            an instance of the backend

        Raises
        ------
        QiskitError
            if the backend can not be instantiated
        """

        # Verify that the backend can be instantiated.
        try:
            backend_instance = backend_cls(provider=self)
        except Exception as err:
            raise QiskitError(
                "Backend %s could not be instantiated: %s" % (backend_cls, err)
            )

        return backend_instance

    def __str__(self):
        return "C3"
