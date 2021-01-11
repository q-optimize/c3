"""
testing module for three wave mixing SNAIL class
"""

import pytest
import numpy as np
from c3.c3objs import Quantity
from c3.system.chip import SNAIL, Coupling, Drive
from c3.system.tasks import InitialiseGround, ConfusionMatrix
from c3.system.model import Model
import c3.libraries.hamiltonians as hamiltonians

qubit_lvls = 3
freq_S = 5e9
anhar_S = -210e6
beta_S = 0.3e9
t1_S = 27e-6
t2star_S = 39e-6
S_temp = 50e-3

S = SNAIL(
    name="Test",
    desc="SNAIL",
    freq=Quantity(value=freq_S, min_val=4.995e9, max_val=5.005e9, unit="Hz 2pi"),
    anhar=Quantity(value=anhar_S, min_val=-380e6, max_val=-120e6, unit="Hz 2pi"),
    beta =Quantity(value=beta_S, min_val=250e6, max_val=350e6, unit="Hz 2pi"),
    hilbert_dim=qubit_lvls,
    t1=Quantity(value=t1_S, min_val=1e-6, max_val=90e-6, unit="s"),
    t2star=Quantity(value=t2star_S, min_val=10e-6, max_val=90e-3, unit="s"),
    temp=Quantity(value=S_temp, min_val=0.0, max_val=0.12, unit="K"),
)

drive = Drive(
    name="d1",
    desc="Drive 1",
    comment="Drive line 1 on SNAIL 1",
    connected=["Test"],
    hamiltonian_func=hamiltonians.y_drive,
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
    [S],  # Individual, self-contained components
    [drive],  # Interactions between components
    [conf_matrix, init_ground],  # SPAM processing
)

model.set_dressed(False)

hdrift, hks = model.get_Hamiltonians()

@pytest.mark.unit
def test_SNAIL_eigenfrequencies_1() -> None:
    "Eigenfrequency of SNAIL"
    assert hdrift[1,1] - hdrift[0,0] == freq_S * 2 * np.pi # for the 0.2dev version, comment out the 2 pi


@pytest.mark.unit
def test_three_wave_mixer_properties() -> None:
    "Test if the values of the three wave mixer element are assigned correctly"

    assert float(model.subsystems["Test"].params["freq"].get_value()) == freq_S* 2 * np.pi
    assert float(model.subsystems["Test"].params["anhar"].get_value()) == anhar_S* 2 * np.pi
    assert float(model.subsystems["Test"].params["t1"].get_value()) == t1_S
    assert float(model.subsystems["Test"].params["beta"].get_value()) == beta_S* 2 * np.pi
    assert float(model.subsystems["Test"].params["temp"].get_value()) == S_temp 
