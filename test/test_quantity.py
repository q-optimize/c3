"""Unit tests for Quantity class"""

import hjson
from c3.c3objs import Quantity

amp = Quantity(value=0.0, min_val=-1.0, max_val=+1.0, unit="V")
amp_dict = {
    'value': 0.0,
    'min_val': -1.0,
    'max_val': 1.0,
    'unit': 'V',
    'symbol': '\\alpha'
}


def test_qty_asdict() -> None:
    assert amp.asdict() == amp_dict


def test_qty_write_cfg() -> None:
    print(hjson.dumps(amp.asdict()))


def test_qty_read_cfg() -> None:
    assert Quantity(**amp_dict).asdict() == amp.asdict()
