from c3.signal.pulse import Envelope, EnvelopeNetZero
from c3.c3objs import Quantity as Qty
from c3.libraries import envelopes
import numpy as np
import pytest

flux_env = Envelope(
    name="flux",
    desc="Flux bias for tunable coupler",
    shape=envelopes.rect,
    params={
        'amp': Qty(
            value=0.1, unit="V"
        ),
        't_final': Qty(
            value=10e-9, unit="s"
        ),
        'freq_offset': Qty(
            value=0, min_val=0, max_val=1, unit='Hz 2pi'
        ),
        'xy_angle': Qty(
            value=0, min_val=0, max_val=np.pi, unit='rad'
        )
    }
)

flux_env_netzero = EnvelopeNetZero(
    name="flux",
    desc="Flux bias for tunable coupler",
    shape=envelopes.rect,
    params={
        'amp': Qty(
            value=0.1, unit="V"
        ),
        't_final': Qty(
            value=5e-9, unit="s"
        ),
        'freq_offset': Qty(
            value=0, min_val=0, max_val=1, unit='Hz 2pi'
        ),
        'xy_angle': Qty(
            value=0, min_val=0, max_val=np.pi, unit='rad'
        )
    }
)


@pytest.mark.envelope
def test_envelope() -> None:
    ts = np.arange(0, 12e-9, 1e-9)
    shape = flux_env.get_shape_values(ts)
    assert np.all(shape.numpy() == np.array([1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 0., 0.]))


@pytest.mark.envelope
def test_envelope_netzero() -> None:
    ts = np.arange(0, 12e-9, 1e-9)
    shape = flux_env_netzero.get_shape_values(ts)
    assert np.all(shape.numpy() == np.array([1.,  1.,  1.,  1.,  1.,  0., -1., -1., -1., -1., -1., -0.]))

