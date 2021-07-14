"""Test module for model learning on a small dataset generated from the
blackbox in examples/Simulated_Calibration.ipynb

PLEASE UPDATE THE DETAILS BELOW IF YOU CHANGE THE DATASET OR THE MODEL

The dataset is an ORBIT experiment with 20 ORBIT sequences
An individual sequence looks like the following:
['ry90p[0]', 'rx90p[0]', 'rx90p[0]', 'rx90m[0]', 'ry90p[0]', 'ry90p[0]', 'rx90p[0]',
'ry90p[0]', 'rx90p[0]', 'rx90p[0]', 'ry90p[0]', 'rx90m[0]', 'rx90p[0]', 'rx90p[0]',
'ry90p[0]', 'ry90p[0]', 'rx90p[0]', 'ry90p[0]', 'ry90m[0]', 'rx90p[0]', 'rx90p[0]',
'ry90m[0]', 'rx90p[0]', 'rx90p[0]', 'rx90p[0]', 'rx90p[0]']

The optimization map is as below:
[['rx90p[0]-d1-gauss-amp', 'ry90p[0]-d1-gauss-amp', 'rx90m[0]-d1-gauss-amp',
'ry90m[0]-d1-gauss-amp'], ['rx90p[0]-d1-gauss-delta', 'ry90p[0]-d1-gauss-delta',
'rx90m[0]-d1-gauss-delta', 'ry90m[0]-d1-gauss-delta'], ['rx90p[0]-d1-gauss-freq_offset',
'ry90p[0]-d1-gauss-freq_offset', 'rx90m[0]-d1-gauss-freq_offset',
'ry90m[0]-d1-gauss-freq_offset'], ['id[0]-d1-carrier-framechange']]

The blackbox for the dataset has a qubit frequency of 5e9 and anharmonicity of -2.10e8
We initiate the model with a qubit frequency of 5.0001e9 and anharmonicity of -2.100001e8
"""

import hjson
import pytest
import numpy as np

from c3.optimizers.modellearning import ModelLearning
from c3.experiment import Experiment

OPT_CONFIG_FILE_NAME = "test/c3.cfg"
DESIRED_PARAMS = [-210000000.0, 5000000000.0]
RELATIVE_TOLERANCE = 1e-3


@pytest.mark.integration
@pytest.mark.slow
def test_model_learning() -> None:
    with open(OPT_CONFIG_FILE_NAME, "r") as cfg_file:
        cfg = hjson.load(cfg_file)

    cfg.pop("optim_type")

    exp = Experiment()
    exp.read_config(cfg.pop("exp_cfg"))
    exp.pmap.set_opt_map(
        [[tuple(par) for par in pset] for pset in cfg.pop("exp_opt_map")]
    )
    opt = ModelLearning(**cfg, pmap=exp.pmap)
    opt.set_exp(exp)
    opt.set_created_by(OPT_CONFIG_FILE_NAME)
    opt.run()
    np.testing.assert_allclose(
        opt.current_best_params, DESIRED_PARAMS, rtol=RELATIVE_TOLERANCE
    )
