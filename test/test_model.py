"""
testing module for Model class
"""

import numpy as np
from c3.c3objs import Quantity
from c3.system.chip import Qubit, Coupling, Drive
from c3.system.tasks import InitialiseGround, ConfusionMatrix
from c3.system.model import Model
import c3.libraries.hamiltonians as hamiltonians

qubit_lvls = 3
freq_q1 = 5e9
anhar_q1 = -210e6
t1_q1 = 27e-6
t2star_q1 = 39e-6
qubit_temp = 50e-3

q1 = Qubit(
    name="Q1",
    desc="Qubit 1",
    freq=Quantity(
        value=freq_q1,
        min_val=4.995e9,
        max_val=5.005e9,
        unit='Hz 2pi'
    ),
    anhar=Quantity(
        value=anhar_q1,
        min_val=-380e6,
        max_val=-120e6,
        unit='Hz 2pi'
    ),
    hilbert_dim=qubit_lvls,
    t1=Quantity(
        value=t1_q1,
        min_val=1e-6,
        max_val=90e-6,
        unit='s'
    ),
    t2star=Quantity(
        value=t2star_q1,
        min_val=10e-6,
        max_val=90e-3,
        unit='s'
    ),
    temp=Quantity(
        value=qubit_temp,
        min_val=0.0,
        max_val=0.12,
        unit='K'
    )
)

freq_q2 = 5.6e9
anhar_q2 = -240e6
t1_q2 = 23e-6
t2star_q2 = 31e-6
q2 = Qubit(
    name="Q2",
    desc="Qubit 2",
    freq=Quantity(
        value=freq_q2,
        min_val=5.595e9,
        max_val=5.605e9,
        unit='Hz 2pi'
    ),
    anhar=Quantity(
        value=anhar_q2,
        min_val=-380e6,
        max_val=-120e6,
        unit='Hz 2pi'
    ),
    hilbert_dim=qubit_lvls,
    t1=Quantity(
        value=t1_q2,
        min_val=1e-6,
        max_val=90e-6,
        unit='s'
    ),
    t2star=Quantity(
        value=t2star_q2,
        min_val=10e-6,
        max_val=90e-6,
        unit='s'
    ),
    temp=Quantity(
        value=qubit_temp,
        min_val=0.0,
        max_val=0.12,
        unit='K'
    )
)

coupling_strength = 20e6
q1q2 = Coupling(
    name="Q1-Q2",
    desc="coupling",
    comment="Coupling qubit 1 to qubit 2",
    connected=["Q1", "Q2"],
    strength=Quantity(
        value=coupling_strength,
        min_val=-1 * 1e3,
        max_val=200e6,
        unit='Hz 2pi'
    ),
    hamiltonian_func=hamiltonians.int_XX
)


drive = Drive(
    name="d1",
    desc="Drive 1",
    comment="Drive line 1 on qubit 1",
    connected=["Q1"],
    hamiltonian_func=hamiltonians.x_drive
)
drive2 = Drive(
    name="d2",
    desc="Drive 2",
    comment="Drive line 2 on qubit 2",
    connected=["Q2"],
    hamiltonian_func=hamiltonians.x_drive
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
    init_temp=Quantity(
        value=init_temp,
        min_val=-0.001,
        max_val=0.22,
        unit='K'
    )
)

model = Model(
    [q1, q2],  # Individual, self-contained components
    [drive, drive2, q1q2],  # Interactions between components
    [conf_matrix, init_ground]  # SPAM processing
)

hdrift, hks = model.get_Hamiltonians()


def test_model_eigenfrequencies_1() -> None:
    "Eigenfrequency of qubit 1"
    assert hdrift[3, 3] - hdrift[0, 0] == freq_q1 * 2 * np.pi


def test_model_eigenfrequencies_2() -> None:
    "Eigenfrequency of qubit 2"
    assert hdrift[1, 1] - hdrift[0, 0] == freq_q2 * 2 * np.pi


def test_model_couplings() -> None:
    assert hks["d1"][3, 0] == 1
    assert hks["d2"][1, 0] == 1
