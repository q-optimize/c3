"""
testing module for Model class
"""
import pickle

import pytest
import copy
import numpy as np
import tensorflow as tf
from c3.c3objs import Quantity
from c3.libraries.chip import Qubit, Coupling, Drive
from c3.libraries.tasks import InitialiseGround, ConfusionMatrix
from c3.model import Model, Model_basis_change
import c3.libraries.hamiltonians as hamiltonians
from c3.parametermap import ParameterMap

qubit_lvls = 3
freq_q1 = 5e9
anhar_q1 = -210e6
t1_q1 = 27e-6
t2star_q1 = 39e-6
qubit_temp = 50e-3

q1 = Qubit(
    name="Q1",
    desc="Qubit 1",
    freq=Quantity(value=freq_q1, min_val=4.995e9, max_val=5.005e9, unit="Hz 2pi"),
    anhar=Quantity(value=anhar_q1, min_val=-380e6, max_val=-120e6, unit="Hz 2pi"),
    hilbert_dim=qubit_lvls,
    t1=Quantity(value=t1_q1, min_val=1e-6, max_val=90e-6, unit="s"),
    t2star=Quantity(value=t2star_q1, min_val=10e-6, max_val=90e-3, unit="s"),
    temp=Quantity(value=qubit_temp, min_val=0.0, max_val=0.12, unit="K"),
)

freq_q2 = 5.6e9
anhar_q2 = -240e6
t1_q2 = 23e-6
t2star_q2 = 31e-6
q2 = Qubit(
    name="Q2",
    desc="Qubit 2",
    freq=Quantity(value=freq_q2, min_val=5.595e9, max_val=5.605e9, unit="Hz 2pi"),
    anhar=Quantity(value=anhar_q2, min_val=-380e6, max_val=-120e6, unit="Hz 2pi"),
    hilbert_dim=qubit_lvls,
    t1=Quantity(value=t1_q2, min_val=1e-6, max_val=90e-6, unit="s"),
    t2star=Quantity(value=t2star_q2, min_val=10e-6, max_val=90e-6, unit="s"),
    temp=Quantity(value=qubit_temp, min_val=0.0, max_val=0.12, unit="K"),
)

coupling_strength = 20e6
q1q2 = Coupling(
    name="Q1-Q2",
    desc="coupling",
    comment="Coupling qubit 1 to qubit 2",
    connected=["Q1", "Q2"],
    strength=Quantity(
        value=coupling_strength, min_val=-1 * 1e3, max_val=200e6, unit="Hz 2pi"
    ),
    hamiltonian_func=hamiltonians.int_XX,
)

drive = Drive(
    name="d1",
    desc="Drive 1",
    comment="Drive line 1 on qubit 1",
    connected=["Q1"],
    hamiltonian_func=hamiltonians.x_drive,
)
drive2 = Drive(
    name="d2",
    desc="Drive 2",
    comment="Drive line 2 on qubit 2",
    connected=["Q2"],
    hamiltonian_func=hamiltonians.x_drive,
)

m00_q1 = 0.97  # Prop to read qubit 1 state 0 as 0
m01_q1 = 0.04  # Prop to read qubit 1 state 0 as 1
m00_q2 = 0.96  # Prop to read qubit 2 state 0 as 0
m01_q2 = 0.05  # Prop to read qubit 2 state 0 as 1
one_zeros = np.array([0] * qubit_lvls)
zero_ones = np.array([1] * qubit_lvls)
one_zeros[0] = 1
zero_ones[0] = 0
val1 = one_zeros * m00_q1 + zero_ones * m01_q1
val2 = one_zeros * m00_q2 + zero_ones * m01_q2
min = one_zeros * 0.8 + zero_ones * 0.0
max = one_zeros * 1.0 + zero_ones * 0.2
confusion_row1 = Quantity(value=val1, min_val=min, max_val=max, unit="")
confusion_row2 = Quantity(value=val2, min_val=min, max_val=max, unit="")
conf_matrix = ConfusionMatrix(Q1=confusion_row1, Q2=confusion_row2)

init_temp = 50e-3
init_ground = InitialiseGround(
    init_temp=Quantity(value=init_temp, min_val=-0.001, max_val=0.22, unit="K")
)

model = Model(
    [q1, q2],  # Individual, self-contained components
    [drive, drive2, q1q2],  # Interactions between components
    [conf_matrix, init_ground],  # SPAM processing
)

pmap = ParameterMap(model=model)
model.set_dressed(False)

hdrift, hks = model.get_Hamiltonians()

with open("test/model.pickle", "rb") as filename:
    test_data = pickle.load(filename)


@pytest.mark.unit
def test_model_eigenfrequencies_1() -> None:
    "Eigenfrequency of qubit 1"
    assert hdrift[3, 3] - hdrift[0, 0] == freq_q1 * 2 * np.pi


@pytest.mark.unit
def test_model_eigenfrequencies_2() -> None:
    "Eigenfrequency of qubit 2"
    assert hdrift[1, 1] - hdrift[0, 0] == freq_q2 * 2 * np.pi


@pytest.mark.unit
def test_model_couplings() -> None:
    assert hks["d1"][3, 0] == 1
    assert hks["d2"][1, 0] == 1


@pytest.mark.unit
def test_model_get_hamiltonian() -> None:
    ham = model.get_Hamiltonian()
    np.testing.assert_allclose(ham, hdrift)

    sig = {"d1": {"ts": np.linspace(0, 5e-9, 10), "values": np.linspace(0e9, 20e9, 10)}}
    hams = model.get_Hamiltonian(sig)
    np.testing.assert_allclose(hams, test_data["sliced_hamiltonians"])


@pytest.mark.unit
def test_get_qubit_frequency() -> None:
    np.testing.assert_allclose(
        model.get_qubit_freqs(), [4999294802.027272, 5600626454.433859]
    )


@pytest.mark.unit
def test_get_indeces() -> None:
    assert model.get_state_index((0, 0)) == 0
    assert model.get_state_index((1, 0)) == 3
    assert model.get_state_index((1, 1)) == 4
    assert model.get_state_index((2, 1)) == 7

    actual = model.get_state_indeces([(0, 0), (1, 0), (2, 0), (1, 1)])
    desired = [0, 3, 6, 4]
    np.testing.assert_equal(actual=actual, desired=desired)


@pytest.mark.unit
def test_model_update_by_parametermap() -> None:
    pmap.set_parameters([freq_q1 * 0.9995], [[("Q1", "freq")]])
    hdrift_a, _ = model.get_Hamiltonians()

    pmap.set_parameters([freq_q1 * 1.0005], [[("Q1", "freq")]])
    hdrift_b, _ = model.get_Hamiltonians()

    assert hdrift_a[3, 3] - hdrift_a[0, 0] == freq_q1 * 0.9995 * 2 * np.pi
    assert hdrift_b[3, 3] - hdrift_b[0, 0] == freq_q1 * 1.0005 * 2 * np.pi


@pytest.mark.unit
def test_model_recompute() -> None:
    """Test whether setting a model parameter triggers recompute."""
    assert not pmap.update_model
    pmap.set_opt_map([[("Q1-Q2", "strength")]])
    assert pmap.update_model


@pytest.mark.unit
def test_model_thermal_state() -> None:
    """Test computation of initial state"""
    model.set_lindbladian(True)
    np.testing.assert_almost_equal(model.get_init_state()[0], 0.9871, decimal=4)


@pytest.mark.unit
def test_model_init_state() -> None:
    """Test computation of initial state"""
    np.testing.assert_almost_equal(model.get_ground_state()[0], 1, decimal=4)


@pytest.mark.unit
def test_model_dressed_basis() -> None:
    """Test dressed basis"""
    mod = copy.deepcopy(model)
    # Set qubits close to resonance to test ordering
    mod.subsystems["Q2"].params["freq"] = Quantity(
        value=freq_q1 + coupling_strength / 10, unit="Hz 2pi"
    )
    mod.set_dressed(True)


# Test model with arbitrary basis

U_transform = tf.cast(
    tf.random.uniform([qubit_lvls**2] * 2, minval=0, maxval=1, seed=0),
    dtype=tf.complex128,
)
model_arb_basis = Model_basis_change(
    [q1, q2],  # Individual, self-contained components
    [drive, drive2, q1q2],  # Interactions between components
    U_transform=U_transform,  # Arbitarry basis change
)


@pytest.mark.unit
def test_transform_applied() -> None:
    """Test wether U_transform is applied correctly"""
    drift_ham = model_arb_basis.drift_ham
    control_hams = model_arb_basis.control_hams
    dressed_drift_ham, dressed_control_hams = model_arb_basis.get_Hamiltonians()

    # Reorder matrix
    signed_rm = tf.cast(
        tf.sign(tf.math.real(U_transform)) * model_arb_basis.reorder_matrix,
        dtype=tf.complex128,
    )

    # Reordered basis transform
    U_ordered = tf.matmul(U_transform, tf.transpose(signed_rm))

    H_test = tf.matmul(tf.matmul(tf.linalg.adjoint(U_ordered), drift_ham), U_ordered)
    np.testing.assert_almost_equal(H_test.numpy(), dressed_drift_ham.numpy())

    for key in control_hams.keys():
        H_test = tf.matmul(
            tf.matmul(tf.linalg.adjoint(U_ordered), control_hams[key]), U_ordered
        )
        np.testing.assert_almost_equal(
            H_test.numpy(), dressed_control_hams[key].numpy()
        )
