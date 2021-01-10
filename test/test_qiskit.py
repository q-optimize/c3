"""Module for testing C3 Integration with Qiskit
"""

from c3.qiskit import C3Provider
from qiskit.providers import BackendV1 as Backend

import pytest


@pytest.mark.unit
@pytest.mark.qiskit
def test_backends():
    """Test backends() function which returns all available backends"""
    c3_qiskit = C3Provider()
    for backend in c3_qiskit.backends():
        assert isinstance(backend, Backend)


@pytest.mark.unit
@pytest.mark.qiskit
@pytest.mark.parametrize("backend", ["c3_qasm_simulator"])
def test_get_backend(backend):
    """Test get_backend() which returns the backend with matching name

    Parameters
    ----------
    backend : str
        name of the backend that is to be fetched
    """
    qtf = C3Provider()
    received_backend = qtf.get_backend(backend)
    assert isinstance(received_backend, Backend)
