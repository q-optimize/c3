import unittest
import numpy as np
from c3po.component import Qubit as Qubit
from c3po.model import Model as mdl

class TestModel(unittest.TestCase):

    def test_qubit(self):
        q1 = Qubit(
                name = "Q1",
                desc = "Qubit 1",
                comment = "",
                freq = 5e9*2*np.pi,
                anhar = -150e6*2*np.pi,
                hilbert_dim = 3
            )
        chip_elements = [q1]
        model = mdl(chip_elements, 0.7e9*2*np.pi)
        self.assertCountEqual(model.params, [5e9*2*np.pi, -150e6*2*np.pi])
        self.assertCountEqual(model.params_desc, [['Q1', 'freq'], ['Q1', 'anhar']])

if __name__ == '__main__':
    unittest.main()
