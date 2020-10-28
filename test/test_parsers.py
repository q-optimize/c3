from c3.utils.parsers import create_model

model = create_model("test/test_model.cfg")


def test_subsystems() -> None:
    assert list(model.subsystems.keys()) == ['Q1', 'Q2', 'Q3', 'Q4', 'Q5', 'Q6']


def test_couplings() -> None:
    assert list(model.couplings.keys()) == ['Q1-Q2', 'Q4-Q6', 'd1', 'd2']


def test_q6_freq() -> None:
    assert str(model.subsystems['Q6'].params['freq']) == '4.600 GHz 2pi '
